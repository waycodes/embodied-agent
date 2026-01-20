"""Index builder for episodes.parquet from compiled outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from embodied_datakit.artifacts import ArtifactLayout
from embodied_datakit.schema.index import EpisodeIndexRecord, get_index_schema


class IndexBuilder:
    """Build episode index from compiled dataset outputs."""
    
    def __init__(self, layout: ArtifactLayout) -> None:
        """Initialize index builder.
        
        Args:
            layout: Artifact layout for the dataset.
        """
        self.layout = layout
        self._records: list[EpisodeIndexRecord] = []
    
    def scan_episodes(self) -> Iterator[EpisodeIndexRecord]:
        """Scan existing episodes parquet and yield records.
        
        Yields:
            EpisodeIndexRecord for each episode.
        """
        episodes_path = self.layout.episodes_index_path
        if not episodes_path.exists():
            return
        
        table = pq.read_table(episodes_path)
        rows = table.to_pydict()
        
        for i in range(len(table)):
            record = EpisodeIndexRecord(
                episode_id=rows["episode_id"][i],
                dataset_name=rows["dataset_name"][i],
                robot_id=rows.get("robot_id", ["unknown"] * len(table))[i],
                task_id=rows.get("task_id", [0] * len(table))[i],
                task_text=rows.get("task_text", [""] * len(table))[i],
                num_steps=rows.get("num_steps", [0] * len(table))[i],
                duration_secs=rows.get("duration_secs", [0.0] * len(table))[i],
                start_timestamp=rows.get("start_timestamp", [0.0] * len(table))[i],
                end_timestamp=rows.get("end_timestamp", [0.0] * len(table))[i],
                camera_set=rows.get("camera_set", [""] * len(table))[i],
                action_space_type=rows.get("action_space_type", ["custom"] * len(table))[i],
                invalid=rows.get("invalid", [False] * len(table))[i],
                source_uri=rows.get("source_uri", [""] * len(table))[i],
                split=rows.get("split", ["train"] * len(table))[i],
                parquet_file=rows.get("parquet_file", [""] * len(table))[i],
                parquet_row_start=rows.get("parquet_row_start", [0] * len(table))[i],
                parquet_row_end=rows.get("parquet_row_end", [0] * len(table))[i],
                video_offsets=rows.get("video_offsets", ["{}"] * len(table))[i],
            )
            yield record
    
    def add_record(self, record: EpisodeIndexRecord) -> None:
        """Add a record to the index."""
        self._records.append(record)
    
    def build(self) -> Path:
        """Build and write the index.
        
        Returns:
            Path to written index file.
        """
        if not self._records:
            # Try to load from existing
            self._records = list(self.scan_episodes())
        
        if not self._records:
            # Write empty table
            table = pa.table({}, schema=get_index_schema())
        else:
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
        
        output_path = self.layout.episodes_index_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path)
        return output_path
    
    def validate_schema(self) -> list[str]:
        """Validate index schema compliance.
        
        Returns:
            List of validation errors (empty if valid).
        """
        errors: list[str] = []
        
        if not self.layout.episodes_index_path.exists():
            errors.append("Index file does not exist")
            return errors
        
        table = pq.read_table(self.layout.episodes_index_path)
        expected_schema = get_index_schema()
        
        # Check all required fields present
        for field in expected_schema:
            if field.name not in table.column_names:
                errors.append(f"Missing required field: {field.name}")
        
        return errors
    
    @property
    def record_count(self) -> int:
        """Number of records in index."""
        return len(self._records)
