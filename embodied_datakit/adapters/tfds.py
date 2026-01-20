"""TFDS/RLDS adapter for Open X-Embodiment and RLDS datasets."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import numpy as np

from embodied_datakit.adapters.base import BaseAdapter
from embodied_datakit.schema.conversion import to_numpy
from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.spec import DatasetSpec, FeatureSpec
from embodied_datakit.schema.step import Step

logger = logging.getLogger(__name__)


def _check_tfds_available() -> None:
    """Check that TensorFlow Datasets is available."""
    try:
        import tensorflow_datasets  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "TensorFlow Datasets is required for RLDS ingestion. "
            "Install with: pip install embodied-datakit[tfds]"
        ) from e


class TFDSAdapter(BaseAdapter):
    """Adapter for loading TFDS-registered RLDS datasets.

    Supports loading by dataset name (e.g., 'berkeley_autolab_ur5')
    or from a pre-built directory.
    """

    def __init__(
        self,
        source_uri: str,
        data_dir: str | None = None,
    ) -> None:
        """Initialize TFDS adapter.

        Args:
            source_uri: Dataset name or path. Can be:
                - TFDS name: "berkeley_autolab_ur5"
                - OXE URI: "oxe://berkeley_autolab_ur5"
                - Directory path: "/path/to/dataset"
            data_dir: Optional TFDS data directory.
        """
        super().__init__(source_uri)
        _check_tfds_available()

        # Parse source URI
        if source_uri.startswith("oxe://"):
            self.dataset_name = source_uri[6:]
        elif source_uri.startswith("tfds://"):
            self.dataset_name = source_uri[7:]
        else:
            self.dataset_name = source_uri

        self.data_dir = data_dir
        self._builder = None
        self._info = None

    def _get_builder(self) -> Any:
        """Get or create TFDS builder."""
        if self._builder is not None:
            return self._builder

        import tensorflow_datasets as tfds

        try:
            self._builder = tfds.builder(self.dataset_name, data_dir=self.data_dir)
            self._info = self._builder.info
        except Exception as e:
            logger.error(f"Failed to load dataset '{self.dataset_name}': {e}")
            raise

        return self._builder

    def probe(self) -> DatasetSpec:
        """Probe the dataset and return its specification."""
        builder = self._get_builder()
        info = builder.info

        # Extract observation and action schemas from steps
        steps_feature = info.features.get("steps")
        if steps_feature is None:
            raise ValueError(f"Dataset {self.dataset_name} has no 'steps' feature (not RLDS?)")

        observation_schema: dict[str, FeatureSpec] = {}
        action_schema: FeatureSpec | None = None
        camera_names: list[str] = []

        # Parse steps feature
        steps_info = steps_feature.feature
        obs_info = steps_info.get("observation")
        if obs_info:
            observation_schema, camera_names = self._parse_observation_schema(obs_info)

        action_info = steps_info.get("action")
        if action_info:
            action_schema = self._parse_feature(action_info, "action")

        # Determine control rate from metadata
        control_rate_hz = 10.0  # Default fallback
        if hasattr(info, "metadata") and info.metadata and "control_frequency_hz" in info.metadata:
            control_rate_hz = float(info.metadata["control_frequency_hz"])

        return DatasetSpec(
            dataset_id=self.dataset_name,
            dataset_name=info.name,
            observation_schema=observation_schema,
            action_schema=action_schema,
            control_rate_hz=control_rate_hz,
            camera_names=camera_names,
            source_uri=self.source_uri,
        )

    def _parse_observation_schema(
        self, obs_info: Any
    ) -> tuple[dict[str, FeatureSpec], list[str]]:
        """Parse observation features to schema."""
        schema: dict[str, FeatureSpec] = {}
        cameras: list[str] = []

        def traverse(info: Any, prefix: str = "observation") -> None:
            if hasattr(info, "keys"):
                # Dict-like feature
                for key in info.keys():
                    child = info[key]
                    child_prefix = f"{prefix}.{key}" if prefix else key
                    traverse(child, child_prefix)
            else:
                # Leaf feature
                spec = self._parse_feature(info, prefix)
                if spec:
                    schema[prefix] = spec
                    # Detect cameras
                    if prefix.startswith("observation.images.") or "image" in prefix.lower():
                        camera_name = prefix.split(".")[-1]
                        if camera_name not in cameras:
                            cameras.append(camera_name)

        traverse(obs_info)
        return schema, cameras

    def _parse_feature(self, feature: Any, name: str) -> FeatureSpec | None:
        """Parse a single TFDS feature to FeatureSpec."""
        if hasattr(feature, "shape") and hasattr(feature, "dtype"):
            shape = tuple(feature.shape) if feature.shape else ()
            dtype_str = feature.dtype.name if hasattr(feature.dtype, "name") else str(feature.dtype)

            # Detect video/image features
            is_video = len(shape) >= 3 and shape[-1] in (1, 3, 4)

            return FeatureSpec(
                dtype=dtype_str,
                shape=shape,
                description=name,
                is_video=is_video,
            )

        return None

    def iter_episodes(
        self,
        split: str = "train",
        selector: str | None = None,
    ) -> Iterator[Episode]:
        """Iterate over episodes from the dataset."""
        import tensorflow_datasets as tfds  # noqa: F401 - used for type

        builder = self._get_builder()

        # Parse selector
        start_idx, end_idx = self.parse_selector(selector)

        # Build split string
        split_str = split
        if start_idx is not None or end_idx is not None:
            start = start_idx or 0
            end = end_idx or ""
            split_str = f"{split}[{start}:{end}]"

        # Load dataset
        try:
            ds = builder.as_dataset(split=split_str)
        except Exception as e:
            logger.warning(f"Failed to load split '{split_str}': {e}")
            split_str = split  # Fallback to full split
            ds = builder.as_dataset(split=split_str)

        # Iterate over episodes
        for episode_idx, episode_data in enumerate(ds):
            episode = self._parse_episode(
                episode_data,
                episode_idx=episode_idx,
            )
            if episode is not None:
                yield episode

    def _parse_episode(
        self,
        episode_data: dict[str, Any],
        episode_idx: int,
    ) -> Episode | None:
        """Parse a single TFDS episode to canonical Episode."""
        # Extract episode-level metadata
        episode_id = f"{self.dataset_name}_{episode_idx:06d}"

        # Get steps
        steps_data = episode_data.get("steps")
        if steps_data is None:
            logger.warning(f"Episode {episode_idx} has no steps")
            return None

        # Parse steps
        steps: list[Step] = []
        step_list = list(steps_data)

        for step_idx, step_data in enumerate(step_list):
            is_first = step_idx == 0
            is_last = step_idx == len(step_list) - 1

            step = self._parse_step(step_data, is_first, is_last, step_idx)
            if step is not None:
                steps.append(step)

        if not steps:
            logger.warning(f"Episode {episode_idx} produced no valid steps")
            return None

        # Extract task text
        task_text = ""
        if "language_instruction" in episode_data:
            task_text = self._extract_string(episode_data["language_instruction"])
        elif len(steps) > 0 and "observation.language" in steps[0].observation:
            task_text = str(steps[0].observation.get("observation.language", ""))

        return Episode(
            episode_id=episode_id,
            dataset_id=self.dataset_name,
            steps=steps,
            task_text=task_text,
        )

    def _parse_step(
        self,
        step_data: dict[str, Any],
        is_first: bool,
        is_last: bool,
        step_idx: int,
    ) -> Step | None:
        """Parse a single step from TFDS format."""
        try:
            # Parse observation
            obs_raw = step_data.get("observation", {})
            observation = self._flatten_and_convert(obs_raw, "observation")

            # Parse action
            action = None
            if "action" in step_data and not is_last:
                action = to_numpy(step_data["action"])
                if isinstance(action, np.ndarray):
                    action = action.astype(np.float32)

            # Parse reward
            reward = None
            if "reward" in step_data:
                reward = float(to_numpy(step_data["reward"]))

            # Parse discount
            discount = None
            if "discount" in step_data:
                discount = float(to_numpy(step_data["discount"]))

            # Parse terminal
            is_terminal = False
            if "is_terminal" in step_data:
                is_terminal = bool(to_numpy(step_data["is_terminal"]))

            # Parse timestamp (estimate from step index if not available)
            timestamp = step_idx * 0.1  # Default 10Hz

            return Step(
                is_first=is_first,
                is_last=is_last,
                is_terminal=is_terminal,
                observation=observation,
                action=action,
                reward=reward,
                discount=discount,
                timestamp=timestamp,
            )

        except Exception as e:
            logger.warning(f"Failed to parse step {step_idx}: {e}")
            return None

    def _flatten_and_convert(
        self, obs_raw: dict[str, Any], prefix: str
    ) -> dict[str, np.ndarray | str | bytes]:
        """Flatten nested observation dict and convert to numpy."""
        result: dict[str, np.ndarray | str | bytes] = {}

        def traverse(data: Any, key_prefix: str) -> None:
            if isinstance(data, dict):
                for k, v in data.items():
                    new_prefix = f"{key_prefix}.{k}" if key_prefix else k
                    traverse(v, new_prefix)
            else:
                # Leaf value
                value = to_numpy(data)
                if value is not None:
                    result[key_prefix] = value

        traverse(obs_raw, prefix)
        return result

    def _extract_string(self, value: Any) -> str:
        """Extract string from various formats."""
        val = to_numpy(value)
        if isinstance(val, bytes):
            return val.decode("utf-8", errors="replace")
        if isinstance(val, np.ndarray) and val.dtype.kind in ("U", "S", "O"):
            return str(val.item()) if val.size == 1 else str(val)
        return str(val)

    def close(self) -> None:
        """Release resources."""
        self._builder = None
        self._info = None


class DirectoryAdapter(BaseAdapter):
    """Adapter for loading RLDS datasets from a directory (builder_from_directory)."""

    def __init__(self, source_uri: str) -> None:
        """Initialize directory adapter.

        Args:
            source_uri: Path to directory containing RLDS dataset.
        """
        super().__init__(source_uri)
        _check_tfds_available()
        self.data_dir = source_uri

    def probe(self) -> DatasetSpec:
        """Probe the dataset."""
        import tensorflow_datasets as tfds

        builder = tfds.builder_from_directory(self.data_dir)
        info = builder.info

        # Similar parsing as TFDSAdapter
        observation_schema: dict[str, FeatureSpec] = {}
        action_schema: FeatureSpec | None = None

        return DatasetSpec(
            dataset_id=info.name,
            dataset_name=info.name,
            observation_schema=observation_schema,
            action_schema=action_schema,
            source_uri=self.source_uri,
        )

    def iter_episodes(
        self,
        split: str = "train",
        selector: str | None = None,
    ) -> Iterator[Episode]:
        """Iterate over episodes."""
        import tensorflow_datasets as tfds

        builder = tfds.builder_from_directory(self.data_dir)

        # Parse selector
        start_idx, end_idx = self.parse_selector(selector)
        split_str = split
        if start_idx is not None or end_idx is not None:
            start = start_idx or 0
            end = end_idx or ""
            split_str = f"{split}[{start}:{end}]"

        ds = builder.as_dataset(split=split_str)

        for episode_idx, episode_data in enumerate(ds):
            # Use TFDSAdapter's parsing logic
            tfds_adapter = TFDSAdapter.__new__(TFDSAdapter)
            tfds_adapter.dataset_name = builder.info.name
            episode = tfds_adapter._parse_episode(episode_data, episode_idx)
            if episode is not None:
                yield episode
