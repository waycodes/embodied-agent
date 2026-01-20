"""Slice materialization for creating subsets from index queries."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq

from embodied_datakit.index.query import QueryEngine, QueryFilter
from embodied_datakit.schema.index import EpisodeIndexRecord


@dataclass
class SliceManifest:
    """Manifest for a materialized slice."""

    slice_id: str
    parent_dataset: str
    parent_build_id: str
    query_filter: dict
    episode_ids: list[str]
    mode: Literal["copy", "view"]

    def to_dict(self) -> dict:
        return {
            "slice_id": self.slice_id,
            "parent_dataset": self.parent_dataset,
            "parent_build_id": self.parent_build_id,
            "query_filter": self.query_filter,
            "episode_ids": self.episode_ids,
            "mode": self.mode,
        }


class SliceMaterializer:
    """Materialize dataset slices from index queries."""

    def __init__(
        self,
        source_path: Path,
        output_path: Path,
        mode: Literal["copy", "view"] = "view",
    ) -> None:
        """Initialize slice materializer.

        Args:
            source_path: Path to source dataset.
            output_path: Path for output slice.
            mode: 'copy' copies data, 'view' creates manifest only.
        """
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.mode = mode

    def materialize(
        self,
        filter: QueryFilter,
        slice_id: str,
        parent_build_id: str = "",
    ) -> SliceManifest:
        """Materialize a slice from query filter.

        Args:
            filter: Query filter for episode selection.
            slice_id: Unique identifier for this slice.
            parent_build_id: Build ID of parent dataset.

        Returns:
            SliceManifest with provenance info.
        """
        index_path = self.source_path / "meta" / "episodes.parquet"
        engine = QueryEngine(index_path)
        records = engine.query(filter)
        episode_ids = [r.episode_id for r in records]

        self.output_path.mkdir(parents=True, exist_ok=True)

        if self.mode == "copy":
            self._copy_episodes(records)
        else:
            self._write_view_index(records)

        manifest = SliceManifest(
            slice_id=slice_id,
            parent_dataset=str(self.source_path),
            parent_build_id=parent_build_id,
            query_filter=self._filter_to_dict(filter),
            episode_ids=episode_ids,
            mode=self.mode,
        )

        manifest_path = self.output_path / "slice_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        return manifest

    def _copy_episodes(self, records: list[EpisodeIndexRecord]) -> None:
        """Copy episode data to output."""
        # Copy relevant parquet files
        parquet_files = {r.parquet_file for r in records if r.parquet_file}
        data_dir = self.output_path / "data"
        data_dir.mkdir(exist_ok=True)

        for pf in parquet_files:
            src = self.source_path / "data" / pf
            if src.exists():
                shutil.copy2(src, data_dir / pf)

        # Write filtered index
        self._write_view_index(records)

    def _write_view_index(self, records: list[EpisodeIndexRecord]) -> None:
        """Write filtered episodes index."""
        import pyarrow as pa
        from embodied_datakit.schema.index import get_index_schema

        meta_dir = self.output_path / "meta"
        meta_dir.mkdir(exist_ok=True)

        if not records:
            table = pa.table({}, schema=get_index_schema())
        else:
            data = {
                "episode_id": [r.episode_id for r in records],
                "dataset_name": [r.dataset_name for r in records],
                "robot_id": [r.robot_id for r in records],
                "task_id": [r.task_id for r in records],
                "task_text": [r.task_text for r in records],
                "num_steps": [r.num_steps for r in records],
                "duration_secs": [r.duration_secs for r in records],
                "start_timestamp": [r.start_timestamp for r in records],
                "end_timestamp": [r.end_timestamp for r in records],
                "camera_set": [r.camera_set for r in records],
                "action_space_type": [r.action_space_type for r in records],
                "invalid": [r.invalid for r in records],
                "source_uri": [r.source_uri for r in records],
                "split": [r.split for r in records],
                "parquet_file": [r.parquet_file for r in records],
                "parquet_row_start": [r.parquet_row_start for r in records],
                "parquet_row_end": [r.parquet_row_end for r in records],
                "video_offsets": [r.video_offsets for r in records],
                "schema_version": [r.schema_version for r in records],
            }
            table = pa.table(data, schema=get_index_schema())

        pq.write_table(table, meta_dir / "episodes.parquet")

    def _filter_to_dict(self, f: QueryFilter) -> dict:
        """Convert filter to serializable dict."""
        return {
            k: v for k, v in {
                "robot_id": f.robot_id,
                "task_id": f.task_id,
                "task_text_regex": f.task_text_regex,
                "min_steps": f.min_steps,
                "max_steps": f.max_steps,
                "camera_set": f.camera_set,
                "action_space_type": f.action_space_type,
                "invalid": f.invalid,
                "split": f.split,
                "dataset_name": f.dataset_name,
            }.items() if v is not None
        }
