"""Query engine for filtering episodes index."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from embodied_datakit.schema.index import EpisodeIndexRecord


@dataclass
class QueryFilter:
    """Filter criteria for episode queries."""

    robot_id: str | list[str] | None = None
    task_id: int | list[int] | None = None
    task_text_regex: str | None = None
    min_steps: int | None = None
    max_steps: int | None = None
    camera_set: str | None = None
    action_space_type: str | None = None
    invalid: bool | None = None
    split: str | None = None
    dataset_name: str | None = None


class QueryEngine:
    """Query engine for episodes.parquet index."""

    def __init__(self, index_path: Path) -> None:
        """Initialize query engine.

        Args:
            index_path: Path to episodes.parquet.
        """
        self.index_path = index_path
        self._table = None

    def _load(self):
        """Lazy load the parquet table."""
        if self._table is None:
            self._table = pq.read_table(self.index_path)

    def query(self, filter: QueryFilter | None = None) -> list[EpisodeIndexRecord]:
        """Query episodes matching filter criteria.

        Args:
            filter: Filter criteria. If None, returns all episodes.

        Returns:
            List of matching EpisodeIndexRecord.
        """
        self._load()
        rows = self._table.to_pydict()
        n = len(self._table)

        results = []
        for i in range(n):
            record = self._row_to_record(rows, i)
            if filter is None or self._matches(record, filter):
                results.append(record)

        return results

    def query_ids(self, filter: QueryFilter | None = None) -> list[str]:
        """Query episode IDs matching filter.

        Args:
            filter: Filter criteria.

        Returns:
            List of episode IDs.
        """
        return [r.episode_id for r in self.query(filter)]

    def count(self, filter: QueryFilter | None = None) -> int:
        """Count episodes matching filter."""
        return len(self.query(filter))

    def _row_to_record(self, rows: dict, i: int) -> EpisodeIndexRecord:
        """Convert row dict to record."""
        return EpisodeIndexRecord(
            episode_id=rows["episode_id"][i],
            dataset_name=rows["dataset_name"][i],
            robot_id=rows.get("robot_id", ["unknown"] * len(rows["episode_id"]))[i],
            task_id=rows.get("task_id", [0] * len(rows["episode_id"]))[i],
            task_text=rows.get("task_text", [""] * len(rows["episode_id"]))[i],
            num_steps=rows.get("num_steps", [0] * len(rows["episode_id"]))[i],
            duration_secs=rows.get("duration_secs", [0.0] * len(rows["episode_id"]))[i],
            start_timestamp=rows.get("start_timestamp", [0.0] * len(rows["episode_id"]))[i],
            end_timestamp=rows.get("end_timestamp", [0.0] * len(rows["episode_id"]))[i],
            camera_set=rows.get("camera_set", [""] * len(rows["episode_id"]))[i],
            action_space_type=rows.get("action_space_type", ["custom"] * len(rows["episode_id"]))[i],
            invalid=rows.get("invalid", [False] * len(rows["episode_id"]))[i],
            source_uri=rows.get("source_uri", [""] * len(rows["episode_id"]))[i],
            split=rows.get("split", ["train"] * len(rows["episode_id"]))[i],
            parquet_file=rows.get("parquet_file", [""] * len(rows["episode_id"]))[i],
            parquet_row_start=rows.get("parquet_row_start", [0] * len(rows["episode_id"]))[i],
            parquet_row_end=rows.get("parquet_row_end", [0] * len(rows["episode_id"]))[i],
            video_offsets=rows.get("video_offsets", ["{}"] * len(rows["episode_id"]))[i],
        )

    def _matches(self, record: EpisodeIndexRecord, f: QueryFilter) -> bool:
        """Check if record matches filter."""
        if f.robot_id is not None:
            ids = [f.robot_id] if isinstance(f.robot_id, str) else f.robot_id
            if record.robot_id not in ids:
                return False

        if f.task_id is not None:
            ids = [f.task_id] if isinstance(f.task_id, int) else f.task_id
            if record.task_id not in ids:
                return False

        if f.task_text_regex is not None:
            if not re.search(f.task_text_regex, record.task_text, re.IGNORECASE):
                return False

        if f.min_steps is not None and record.num_steps < f.min_steps:
            return False

        if f.max_steps is not None and record.num_steps > f.max_steps:
            return False

        if f.camera_set is not None and f.camera_set not in record.camera_set:
            return False

        if f.action_space_type is not None and record.action_space_type != f.action_space_type:
            return False

        if f.invalid is not None and record.invalid != f.invalid:
            return False

        if f.split is not None and record.split != f.split:
            return False

        if f.dataset_name is not None and record.dataset_name != f.dataset_name:
            return False

        return True
