"""TFRecord shard writer for RLDS export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.spec import DatasetSpec
from embodied_datakit.writers.rlds_tfds.schema import build_rlds_schema

# TensorFlow is optional
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


class TFRecordShardWriter:
    """Write episodes to TFRecord shards for TFDS compatibility."""
    
    def __init__(
        self,
        output_dir: Path | str,
        episodes_per_shard: int = 100,
        split: str = "train",
    ) -> None:
        """Initialize TFRecord shard writer.
        
        Args:
            output_dir: Output directory for TFRecord files.
            episodes_per_shard: Number of episodes per shard file.
            split: Dataset split name.
        """
        if not HAS_TF:
            raise ImportError("TensorFlow required for TFRecord export")
        
        self.output_dir = Path(output_dir)
        self.episodes_per_shard = episodes_per_shard
        self.split = split
        
        self._shard_idx = 0
        self._episode_count = 0
        self._current_writer: tf.io.TFRecordWriter | None = None
        self._shard_episode_count = 0
    
    def _get_shard_path(self) -> Path:
        """Get path for current shard."""
        shard_dir = self.output_dir / self.split
        shard_dir.mkdir(parents=True, exist_ok=True)
        return shard_dir / f"{self.split}-{self._shard_idx:05d}.tfrecord"
    
    def _start_new_shard(self) -> None:
        """Start a new shard file."""
        if self._current_writer is not None:
            self._current_writer.close()
        
        self._current_writer = tf.io.TFRecordWriter(str(self._get_shard_path()))
        self._shard_episode_count = 0
    
    def write_episode(self, episode: Episode, spec: DatasetSpec) -> None:
        """Write an episode to TFRecord.
        
        Args:
            episode: Episode to write.
            spec: Dataset specification.
        """
        # Start new shard if needed
        if (
            self._current_writer is None
            or self._shard_episode_count >= self.episodes_per_shard
        ):
            if self._current_writer is not None:
                self._shard_idx += 1
            self._start_new_shard()
        
        # Serialize episode
        example = self._episode_to_example(episode)
        self._current_writer.write(example.SerializeToString())
        
        self._shard_episode_count += 1
        self._episode_count += 1
    
    def _episode_to_example(self, episode: Episode) -> "tf.train.Example":
        """Convert episode to TF Example."""
        # Build steps as sequence
        steps_features = []
        for step in episode.steps:
            step_features = self._step_to_features(step)
            steps_features.append(step_features)
        
        # Create SequenceExample for variable-length steps
        context = tf.train.Features(feature={
            "episode_id": _bytes_feature(episode.episode_id.encode()),
            "num_steps": _int64_feature(len(episode.steps)),
        })
        
        # Flatten steps into feature lists
        feature_lists = {}
        if steps_features:
            for key in steps_features[0]:
                values = [sf[key] for sf in steps_features]
                feature_lists[key] = tf.train.FeatureList(feature=values)
        
        sequence_example = tf.train.SequenceExample(
            context=context,
            feature_lists=tf.train.FeatureLists(feature_list=feature_lists),
        )
        
        # Convert to Example (simpler format)
        return tf.train.Example(features=tf.train.Features(feature={
            "episode_id": _bytes_feature(episode.episode_id.encode()),
            "serialized_steps": _bytes_feature(sequence_example.SerializeToString()),
        }))
    
    def _step_to_features(self, step: "Step") -> dict[str, "tf.train.Feature"]:
        """Convert step to feature dict."""
        from embodied_datakit.schema.step import Step
        
        features = {
            "is_first": _int64_feature(int(step.is_first)),
            "is_last": _int64_feature(int(step.is_last)),
            "is_terminal": _int64_feature(int(step.is_terminal)),
            "reward": _float_feature(step.reward or 0.0),
            "discount": _float_feature(step.discount or 1.0),
        }
        
        # Add action
        if step.action is not None:
            features["action"] = _float_list_feature(step.action.flatten().tolist())
        
        # Add observations
        for key, value in step.observation.items():
            safe_key = key.replace(".", "_")
            if isinstance(value, np.ndarray):
                if value.dtype == np.uint8:
                    features[f"obs_{safe_key}"] = _bytes_feature(value.tobytes())
                else:
                    features[f"obs_{safe_key}"] = _float_list_feature(value.flatten().tolist())
            elif isinstance(value, (str, bytes)):
                if isinstance(value, str):
                    value = value.encode()
                features[f"obs_{safe_key}"] = _bytes_feature(value)
        
        return features
    
    def finish(self) -> dict[str, Any]:
        """Finish writing and return metadata."""
        if self._current_writer is not None:
            self._current_writer.close()
            self._current_writer = None
        
        return {
            "num_episodes": self._episode_count,
            "num_shards": self._shard_idx + 1,
            "split": self.split,
        }
    
    def write_tfds_metadata(self, spec: DatasetSpec) -> Path:
        """Write TFDS metadata files.
        
        Args:
            spec: Dataset specification.
        
        Returns:
            Path to metadata directory.
        """
        # Write dataset_info.json
        schema = build_rlds_schema(spec)
        info = {
            "name": spec.dataset_name,
            "version": spec.edk_schema_version,
            "description": f"Exported from EmbodiedDataKit: {spec.dataset_id}",
            "features": schema,
            "splits": {
                self.split: {
                    "num_examples": self._episode_count,
                    "num_shards": self._shard_idx + 1,
                }
            },
        }
        
        info_path = self.output_dir / "dataset_info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        
        return info_path


def _bytes_feature(value: bytes) -> "tf.train.Feature":
    """Create bytes feature."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int) -> "tf.train.Feature":
    """Create int64 feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value: float) -> "tf.train.Feature":
    """Create float feature."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_list_feature(values: list[float]) -> "tf.train.Feature":
    """Create float list feature."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))
