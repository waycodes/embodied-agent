"""LeRobot v3 dataset writer.

Writes datasets in LeRobotDataset v3 format:
- meta/info.json: Dataset metadata
- meta/tasks.jsonl: Task descriptions
- meta/stats.json: Feature statistics
- meta/episodes/*.parquet: Episode metadata
- data/chunk-*/episode_*.parquet: Step data
- videos/chunk-*/episode_*_*.mp4: Video shards (optional)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.spec import DatasetSpec
from embodied_datakit.schema.stats import StatsAccumulator
from embodied_datakit.writers.base import BaseWriter

logger = logging.getLogger(__name__)


class LeRobotV3Writer(BaseWriter):
    """Writer for LeRobotDataset v3 format."""

    def __init__(
        self,
        episodes_per_chunk: int = 1000,
        rows_per_parquet: int = 10000,
    ) -> None:
        """Initialize LeRobot v3 writer.

        Args:
            episodes_per_chunk: Episodes per chunk directory.
            rows_per_parquet: Target rows per parquet file.
        """
        super().__init__()
        self.episodes_per_chunk = episodes_per_chunk
        self.rows_per_parquet = rows_per_parquet

        self._stats = StatsAccumulator()
        self._tasks: dict[str, int] = {}
        self._episode_records: list[dict[str, Any]] = []
        self._step_buffer: list[dict[str, Any]] = []
        self._total_steps: int = 0
        self._chunk_idx: int = 0
        self._parquet_idx: int = 0

    def begin(self, spec: DatasetSpec, output_dir: Path) -> None:
        """Begin writing a new dataset."""
        super().begin(spec, output_dir)

        # Create directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "meta").mkdir(exist_ok=True)
        (self.output_dir / "meta" / "episodes").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)

        self._stats = StatsAccumulator()
        self._tasks = {}
        self._episode_records = []
        self._step_buffer = []
        self._total_steps = 0
        self._chunk_idx = 0
        self._parquet_idx = 0

    def write_episode(self, episode: Episode) -> None:
        """Write a single episode."""
        super().write_episode(episode)

        # Track task
        task_id = self._get_or_create_task(episode.task_text)

        # Create episode record
        episode_record = {
            "episode_index": self._episode_count - 1,
            "tasks": [task_id],
            "length": len(episode.steps),
        }
        self._episode_records.append(episode_record)

        # Process steps
        for step_idx, step in enumerate(episode.steps):
            step_record = self._step_to_record(
                step,
                episode_index=self._episode_count - 1,
                frame_index=self._total_steps,
                step_index=step_idx,
            )
            self._step_buffer.append(step_record)
            self._total_steps += 1

            # Update stats
            self._update_stats(step)

        # Flush if buffer is large enough
        if len(self._step_buffer) >= self.rows_per_parquet:
            self._flush_step_buffer()

        # Check if we need a new chunk
        if self._episode_count % self.episodes_per_chunk == 0:
            self._chunk_idx += 1

    def finalize(self) -> list[Path]:
        """Finalize the dataset and return produced artifacts."""
        # Flush remaining steps
        if self._step_buffer:
            self._flush_step_buffer()

        # Write metadata files
        self._write_info_json()
        self._write_tasks_jsonl()
        self._write_stats_json()
        self._write_episodes_parquet()

        return super().finalize()

    def _get_or_create_task(self, task_text: str) -> int:
        """Get or create task ID for task text."""
        if task_text not in self._tasks:
            self._tasks[task_text] = len(self._tasks)
        return self._tasks[task_text]

    def _step_to_record(
        self,
        step: "Step",
        episode_index: int,
        frame_index: int,
        step_index: int,
    ) -> dict[str, Any]:
        """Convert step to flat record for parquet."""
        from embodied_datakit.schema.step import Step

        record: dict[str, Any] = {
            "episode_index": episode_index,
            "frame_index": frame_index,
            "timestamp": step.timestamp,
            "is_first": step.is_first,
            "is_last": step.is_last,
            "is_terminal": step.is_terminal,
        }

        # Add action
        if step.action is not None:
            record["action"] = step.action.tolist()

        # Add observations (flatten nested keys)
        for key, value in step.observation.items():
            col_name = key.replace(".", "_")
            if isinstance(value, np.ndarray):
                # Store images as paths (video encoding would go here)
                if "image" in key.lower() and len(value.shape) == 3:
                    record[col_name] = None  # Placeholder for video path
                else:
                    record[col_name] = value.tolist()
            elif isinstance(value, (str, bytes)):
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                record[col_name] = value
            else:
                record[col_name] = value

        return record

    def _update_stats(self, step: "Step") -> None:
        """Update running statistics with step data."""
        from embodied_datakit.schema.step import Step

        if step.action is not None:
            self._stats.add("action", step.action)

        for key, value in step.observation.items():
            if isinstance(value, np.ndarray) and value.dtype.kind in ("f", "i", "u"):
                if "image" not in key.lower():
                    self._stats.add(key, value)

    def _flush_step_buffer(self) -> None:
        """Flush step buffer to parquet file."""
        if not self._step_buffer:
            return

        chunk_dir = self.output_dir / "data" / f"chunk-{self._chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = chunk_dir / f"steps_{self._parquet_idx:05d}.parquet"

        # Convert to table
        table = pa.Table.from_pylist(self._step_buffer)
        pq.write_table(table, parquet_path)

        self._artifacts.append(parquet_path)
        self._step_buffer = []
        self._parquet_idx += 1

    def _write_info_json(self) -> None:
        """Write meta/info.json."""
        info = {
            "codebase_version": "v3.0",
            "robot_type": self.spec.dataset_name if self.spec else "unknown",
            "fps": self.spec.control_rate_hz if self.spec else 10.0,
            "total_episodes": self._episode_count,
            "total_frames": self._total_steps,
            "total_tasks": len(self._tasks),
            "splits": {"train": list(range(self._episode_count))},
            "data_path": "data/chunk-{chunk:03d}/steps_{file:05d}.parquet",
            "video_path": None,
            "features": self._get_feature_info(),
        }

        info_path = self.output_dir / "meta" / "info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        self._artifacts.append(info_path)

    def _write_tasks_jsonl(self) -> None:
        """Write meta/tasks.jsonl."""
        tasks_path = self.output_dir / "meta" / "tasks.jsonl"
        with open(tasks_path, "w") as f:
            for task_text, task_id in sorted(self._tasks.items(), key=lambda x: x[1]):
                f.write(json.dumps({"task_index": task_id, "task": task_text}) + "\n")
        self._artifacts.append(tasks_path)

    def _write_stats_json(self) -> None:
        """Write meta/stats.json."""
        stats_path = self.output_dir / "meta" / "stats.json"
        dataset_stats = self._stats.compute()
        stats_dict = dataset_stats.to_dict()
        with open(stats_path, "w") as f:
            json.dump(stats_dict, f, indent=2)
        self._artifacts.append(stats_path)

    def _write_episodes_parquet(self) -> None:
        """Write meta/episodes/episodes.parquet."""
        if not self._episode_records:
            return

        episodes_path = self.output_dir / "meta" / "episodes" / "episodes.parquet"
        table = pa.Table.from_pylist(self._episode_records)
        pq.write_table(table, episodes_path)
        self._artifacts.append(episodes_path)

    def _get_feature_info(self) -> dict[str, Any]:
        """Get feature information for info.json."""
        features = {}

        if self.spec:
            # Action
            if self.spec.action_schema:
                features["action"] = {
                    "dtype": self.spec.action_schema.dtype,
                    "shape": list(self.spec.action_schema.shape),
                }

            # Observations
            for key, feat in self.spec.observation_schema.items():
                col_name = key.replace(".", "_")
                features[col_name] = {
                    "dtype": feat.dtype,
                    "shape": list(feat.shape),
                    "is_video": feat.is_video,
                }

        return features
