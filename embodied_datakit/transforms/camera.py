"""Camera selection and canonicalization transforms."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.spec import DatasetSpec
from embodied_datakit.schema.step import Step
from embodied_datakit.transforms.base import BaseTransform

logger = logging.getLogger(__name__)


class SelectCameraTransform(BaseTransform):
    """Select a canonical camera view from available cameras.

    Renames the selected camera to `observation.images.canonical` for
    consistent downstream processing.
    """

    def __init__(
        self,
        camera_name: str | None = None,
        fallback_order: list[str] | None = None,
        target_key: str = "observation.images.canonical",
    ) -> None:
        """Initialize camera selection transform.

        Args:
            camera_name: Specific camera to select. If None, uses fallback_order.
            fallback_order: Priority order for camera selection.
            target_key: Key to store selected camera under.
        """
        super().__init__("select_camera")
        self.camera_name = camera_name
        self.fallback_order = fallback_order or [
            "front",
            "cam_high",
            "image",
            "rgb",
            "agentview_rgb",
            "image_0",
            "exterior_image_1_left",
        ]
        self.target_key = target_key
        self._selected_camera: str | None = None

    def transform_episode(self, episode: Episode, spec: DatasetSpec) -> Episode:
        """Transform episode by selecting canonical camera."""
        # Determine which camera to use
        camera = self._find_camera(episode, spec)
        if camera is None:
            logger.warning(f"No camera found for episode {episode.episode_id}")
            return episode

        self._selected_camera = camera

        # Transform each step
        new_steps = []
        for step in episode.steps:
            new_step = self._transform_step(step, camera)
            new_steps.append(new_step)

        return Episode(
            episode_id=episode.episode_id,
            dataset_id=episode.dataset_id,
            steps=new_steps,
            task_id=episode.task_id,
            task_text=episode.task_text,
            invalid=episode.invalid,
            episode_metadata=episode.episode_metadata,
        )

    def _find_camera(self, episode: Episode, spec: DatasetSpec) -> str | None:
        """Find the best camera to use."""
        # If specific camera requested
        if self.camera_name:
            return self.camera_name

        # Get available cameras from spec or first step
        available_cameras: set[str] = set()

        if spec.camera_names:
            available_cameras.update(spec.camera_names)
        elif episode.steps:
            # Infer from observation keys
            for key in episode.steps[0].observation:
                if key.startswith("observation.images."):
                    cam_name = key.split(".")[-1]
                    available_cameras.add(cam_name)

        # Try fallback order
        for cam in self.fallback_order:
            if cam in available_cameras:
                return cam

        # Return first available
        if available_cameras:
            return sorted(available_cameras)[0]

        return None

    def _transform_step(self, step: Step, camera: str) -> Step:
        """Add canonical camera to step."""
        source_key = f"observation.images.{camera}"
        new_obs = dict(step.observation)

        if source_key in new_obs:
            new_obs[self.target_key] = new_obs[source_key]

        return Step(
            is_first=step.is_first,
            is_last=step.is_last,
            is_terminal=step.is_terminal,
            observation=new_obs,
            action=step.action,
            reward=step.reward,
            discount=step.discount,
            timestamp=step.timestamp,
            step_metadata=step.step_metadata,
        )


class ResizeImagesTransform(BaseTransform):
    """Resize images to target resolution."""

    def __init__(
        self,
        target_size: tuple[int, int] = (256, 256),
        keys: list[str] | None = None,
        interpolation: str = "bilinear",
    ) -> None:
        """Initialize image resize transform.

        Args:
            target_size: Target (height, width).
            keys: Specific image keys to resize. If None, resizes all images.
            interpolation: Interpolation method (bilinear, nearest, lanczos).
        """
        super().__init__("resize_images")
        self.target_size = target_size
        self.keys = keys
        self.interpolation = interpolation

    def transform_episode(self, episode: Episode, spec: DatasetSpec) -> Episode:
        """Transform episode by resizing images."""
        new_steps = []
        for step in episode.steps:
            new_step = self._transform_step(step)
            new_steps.append(new_step)

        return Episode(
            episode_id=episode.episode_id,
            dataset_id=episode.dataset_id,
            steps=new_steps,
            task_id=episode.task_id,
            task_text=episode.task_text,
            invalid=episode.invalid,
            episode_metadata=episode.episode_metadata,
        )

    def _transform_step(self, step: Step) -> Step:
        """Resize images in step."""
        new_obs = dict(step.observation)

        for key, value in step.observation.items():
            if not self._should_resize(key, value):
                continue

            if isinstance(value, np.ndarray) and len(value.shape) == 3:
                resized = self._resize_image(value)
                new_obs[key] = resized

        return Step(
            is_first=step.is_first,
            is_last=step.is_last,
            is_terminal=step.is_terminal,
            observation=new_obs,
            action=step.action,
            reward=step.reward,
            discount=step.discount,
            timestamp=step.timestamp,
            step_metadata=step.step_metadata,
        )

    def _should_resize(self, key: str, value: Any) -> bool:
        """Check if this key should be resized."""
        if self.keys is not None:
            return key in self.keys

        # Auto-detect image keys
        return "image" in key.lower() and isinstance(value, np.ndarray)

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize a single image."""
        # Try using PIL for better quality
        try:
            from PIL import Image

            pil_image = Image.fromarray(image)
            h, w = self.target_size
            resized = pil_image.resize((w, h), Image.BILINEAR)
            return np.array(resized)
        except ImportError:
            # Fallback to simple numpy resize
            return self._numpy_resize(image)

    def _numpy_resize(self, image: np.ndarray) -> np.ndarray:
        """Simple numpy resize (nearest neighbor)."""
        h_target, w_target = self.target_size
        h_src, w_src = image.shape[:2]

        # Create output array
        if len(image.shape) == 3:
            output = np.zeros((h_target, w_target, image.shape[2]), dtype=image.dtype)
        else:
            output = np.zeros((h_target, w_target), dtype=image.dtype)

        # Nearest neighbor sampling
        row_indices = (np.arange(h_target) * h_src / h_target).astype(int)
        col_indices = (np.arange(w_target) * w_src / w_target).astype(int)

        for i, row in enumerate(row_indices):
            for j, col in enumerate(col_indices):
                output[i, j] = image[row, col]

        return output
