"""Episodes metadata table writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.index import EpisodeIndexRecord, get_index_schema
from embodied_datakit.schema.spec import DatasetSpec
from embodied_datakit.writers.video import VideoOffset


class EpisodesTableWriter:
    """Write episodes metadata to Parquet table."""
    
    def __init__(self, output_path: Path | str) -> None:
        """Initialize episodes table writer.
        
        Args:
            output_path: Path to output Parquet file.
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[EpisodeIndexRecord] = []
        self._parquet_row_counter = 0
    
    def add_episode(
        self,
        episode: Episode,
        spec: DatasetSpec,
        parquet_file: str,
        parquet_row_start: int,
        parquet_row_end: int,
        video_offsets: dict[str, VideoOffset] | None = None,
    ) -> EpisodeIndexRecord:
        """Add an episode to the metadata table.
        
        Args:
            episode: Episode to add.
            spec: Dataset specification.
            parquet_file: Path to Parquet shard containing this episode.
            parquet_row_start: Starting row in Parquet.
            parquet_row_end: Ending row in Parquet.
            video_offsets: Video offsets per camera.
        
        Returns:
            Created EpisodeIndexRecord.
        """
        # Build video offsets JSON
        video_offsets_json = "{}"
        if video_offsets:
            video_offsets_json = json.dumps({
                cam: off.to_dict() for cam, off in video_offsets.items()
            })
        
        # Get timestamps
        start_ts = episode.steps[0].timestamp if episode.steps else 0.0
        end_ts = episode.steps[-1].timestamp if episode.steps else 0.0
        
        # Get camera set
        camera_set = ",".join(sorted(episode.get_camera_names()))
        
        record = EpisodeIndexRecord(
            episode_id=episode.episode_id,
            dataset_name=spec.dataset_name,
            robot_id=episode.episode_metadata.get("robot_id", "unknown"),
            task_id=episode.task_id,
            task_text=episode.task_text,
            num_steps=len(episode.steps),
            duration_secs=episode.duration,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            camera_set=camera_set,
            action_space_type=spec.action_space_type,
            invalid=episode.invalid,
            source_uri=spec.source_uri,
            split="train",  # Default, can be updated later
            parquet_file=parquet_file,
            parquet_row_start=parquet_row_start,
            parquet_row_end=parquet_row_end,
            video_offsets=video_offsets_json,
        )
        
        self._records.append(record)
        return record
    
    def write(self) -> Path:
        """Write all records to Parquet file.
        
        Returns:
            Path to written file.
        """
        if not self._records:
            # Write empty table with schema
            table = pa.table({}, schema=get_index_schema())
        else:
            # Convert records to table
            data = {
                "episode_id": [r.episode_id for r in self._records],
                "dataset_name": [r.dataset_name for r in self._records],
                "robot_id": [r.robot_id for r in self._records],
                "task_id": [r.task_id for r in self._records],
                "task_text": [r.task_text for r in self._records],
                "num_steps": [r.num_steps for r in self._records],
                "duration_secs": [r.duration_secs for r in self._records],
                "start_timestamp": [r.start_timestamp for r in self._records],
                "end_timestamp": [r.end_timestamp for r in self._records],
                "camera_set": [r.camera_set for r in self._records],
                "action_space_type": [r.action_space_type for r in self._records],
                "invalid": [r.invalid for r in self._records],
                "source_uri": [r.source_uri for r in self._records],
                "split": [r.split for r in self._records],
                "parquet_file": [r.parquet_file for r in self._records],
                "parquet_row_start": [r.parquet_row_start for r in self._records],
                "parquet_row_end": [r.parquet_row_end for r in self._records],
                "video_offsets": [r.video_offsets for r in self._records],
                "schema_version": [r.schema_version for r in self._records],
            }
            table = pa.table(data, schema=get_index_schema())
        
        pq.write_table(table, self.output_path)
        return self.output_path
    
    @property
    def record_count(self) -> int:
        """Number of records added."""
        return len(self._records)
    
    @property
    def records(self) -> list[EpisodeIndexRecord]:
        """Get all records."""
        return self._records
