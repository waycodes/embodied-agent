"""EpisodeIndexRecord schema for sliceable metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa


@dataclass
class EpisodeIndexRecord:
    """Index record for an episode, enabling query and slicing.

    Attributes:
        episode_id: Globally unique episode ID.
        dataset_name: Source dataset name.
        robot_id: Robot identifier.
        task_id: Integer task ID.
        task_text: Task description text.
        num_steps: Number of steps in episode.
        duration_secs: Episode duration.
        start_timestamp: First step timestamp.
        end_timestamp: Last step timestamp.
        camera_set: Comma-separated camera names.
        action_space_type: Action space type string.
        invalid: Whether episode is marked invalid.
        source_uri: Original source location.
        split: Assigned split (train/val/test).
        parquet_file: Path to Parquet shard.
        parquet_row_start: Starting row in Parquet.
        parquet_row_end: Ending row in Parquet.
        video_offsets: JSON string of camera -> (file, start_frame, num_frames).
    """

    episode_id: str
    dataset_name: str
    robot_id: str = "unknown"
    task_id: int = 0
    task_text: str = ""
    num_steps: int = 0
    duration_secs: float = 0.0
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0
    camera_set: str = ""
    action_space_type: str = "custom"
    invalid: bool = False
    source_uri: str = ""
    split: str = "train"
    parquet_file: str = ""
    parquet_row_start: int = 0
    parquet_row_end: int = 0
    video_offsets: str = "{}"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary for Parquet."""
        return {
            "episode_id": self.episode_id,
            "dataset_name": self.dataset_name,
            "robot_id": self.robot_id,
            "task_id": self.task_id,
            "task_text": self.task_text,
            "num_steps": self.num_steps,
            "duration_secs": self.duration_secs,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "camera_set": self.camera_set,
            "action_space_type": self.action_space_type,
            "invalid": self.invalid,
            "source_uri": self.source_uri,
            "split": self.split,
            "parquet_file": self.parquet_file,
            "parquet_row_start": self.parquet_row_start,
            "parquet_row_end": self.parquet_row_end,
            "video_offsets": self.video_offsets,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeIndexRecord":
        """Create from dictionary."""
        return cls(
            episode_id=data["episode_id"],
            dataset_name=data["dataset_name"],
            robot_id=data.get("robot_id", "unknown"),
            task_id=data.get("task_id", 0),
            task_text=data.get("task_text", ""),
            num_steps=data.get("num_steps", 0),
            duration_secs=data.get("duration_secs", 0.0),
            start_timestamp=data.get("start_timestamp", 0.0),
            end_timestamp=data.get("end_timestamp", 0.0),
            camera_set=data.get("camera_set", ""),
            action_space_type=data.get("action_space_type", "custom"),
            invalid=data.get("invalid", False),
            source_uri=data.get("source_uri", ""),
            split=data.get("split", "train"),
            parquet_file=data.get("parquet_file", ""),
            parquet_row_start=data.get("parquet_row_start", 0),
            parquet_row_end=data.get("parquet_row_end", 0),
            video_offsets=data.get("video_offsets", "{}"),
        )


def get_index_schema() -> pa.Schema:
    """Get PyArrow schema for episode index."""
    return pa.schema([
        pa.field("episode_id", pa.string()),
        pa.field("dataset_name", pa.string()),
        pa.field("robot_id", pa.string()),
        pa.field("task_id", pa.int32()),
        pa.field("task_text", pa.string()),
        pa.field("num_steps", pa.int32()),
        pa.field("duration_secs", pa.float64()),
        pa.field("start_timestamp", pa.float64()),
        pa.field("end_timestamp", pa.float64()),
        pa.field("camera_set", pa.string()),
        pa.field("action_space_type", pa.string()),
        pa.field("invalid", pa.bool_()),
        pa.field("source_uri", pa.string()),
        pa.field("split", pa.string()),
        pa.field("parquet_file", pa.string()),
        pa.field("parquet_row_start", pa.int64()),
        pa.field("parquet_row_end", pa.int64()),
        pa.field("video_offsets", pa.string()),
    ])
