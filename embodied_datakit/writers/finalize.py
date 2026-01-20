"""Dataset finalization utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from embodied_datakit.artifacts import ArtifactLayout
from embodied_datakit.manifest import RunManifest


def compute_file_checksum(path: Path, algorithm: str = "sha256") -> str:
    """Compute checksum of a file.
    
    Args:
        path: Path to file.
        algorithm: Hash algorithm (sha256, md5).
    
    Returns:
        Hex digest of file contents.
    """
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dataset_checksums(layout: ArtifactLayout) -> dict[str, str]:
    """Compute checksums for all dataset files.
    
    Args:
        layout: Artifact layout.
    
    Returns:
        Dict mapping relative paths to checksums.
    """
    checksums: dict[str, str] = {}
    
    # Meta files
    for path in layout.meta_dir.glob("*"):
        if path.is_file():
            rel_path = path.relative_to(layout.root)
            checksums[str(rel_path)] = compute_file_checksum(path)
    
    # Data files
    for path in layout.data_dir.rglob("*.parquet"):
        rel_path = path.relative_to(layout.root)
        checksums[str(rel_path)] = compute_file_checksum(path)
    
    # Video files
    for path in layout.videos_dir.rglob("*.mp4"):
        rel_path = path.relative_to(layout.root)
        checksums[str(rel_path)] = compute_file_checksum(path)
    
    return checksums


class DatasetFinalizer:
    """Finalize a compiled dataset with checksums and manifest."""
    
    def __init__(self, layout: ArtifactLayout) -> None:
        """Initialize finalizer.
        
        Args:
            layout: Artifact layout for the dataset.
        """
        self.layout = layout
    
    def verify_offsets(self, episodes_count: int, expected_steps: int) -> list[str]:
        """Verify episode offsets are consistent.
        
        Args:
            episodes_count: Expected number of episodes.
            expected_steps: Expected total steps.
        
        Returns:
            List of error messages (empty if valid).
        """
        errors: list[str] = []
        
        # Check episodes.parquet exists
        if not self.layout.episodes_index_path.exists():
            errors.append(f"Missing episodes index: {self.layout.episodes_index_path}")
            return errors
        
        # Load and verify
        import pyarrow.parquet as pq
        table = pq.read_table(self.layout.episodes_index_path)
        
        if len(table) != episodes_count:
            errors.append(f"Episode count mismatch: {len(table)} != {episodes_count}")
        
        # Verify parquet row ranges don't overlap
        rows = table.to_pydict()
        ranges = list(zip(rows["parquet_row_start"], rows["parquet_row_end"]))
        ranges.sort()
        
        for i in range(1, len(ranges)):
            if ranges[i][0] < ranges[i-1][1]:
                errors.append(f"Overlapping parquet ranges at index {i}")
        
        return errors
    
    def compute_checksums(self) -> dict[str, str]:
        """Compute checksums for all files."""
        return compute_dataset_checksums(self.layout)
    
    def write_checksums(self, checksums: dict[str, str]) -> Path:
        """Write checksums to file.
        
        Args:
            checksums: Dict of path -> checksum.
        
        Returns:
            Path to checksums file.
        """
        checksums_path = self.layout.meta_dir / "checksums.json"
        with open(checksums_path, "w") as f:
            json.dump(checksums, f, indent=2, sort_keys=True)
        return checksums_path
    
    def seal_manifest(
        self,
        manifest: RunManifest,
        episode_count: int,
        duration_secs: float,
    ) -> Path:
        """Seal and write the build manifest.
        
        Args:
            manifest: RunManifest to seal.
            episode_count: Number of episodes.
            duration_secs: Build duration.
        
        Returns:
            Path to manifest file.
        """
        # Add all artifacts
        for path in self.layout.meta_dir.glob("*"):
            if path.is_file():
                manifest.add_artifact(str(path.relative_to(self.layout.root)))
        
        for path in self.layout.data_dir.rglob("*.parquet"):
            manifest.add_artifact(str(path.relative_to(self.layout.root)))
        
        for path in self.layout.videos_dir.rglob("*.mp4"):
            manifest.add_artifact(str(path.relative_to(self.layout.root)))
        
        # Mark complete
        manifest.complete(episode_count, duration_secs)
        
        # Write
        manifest_path = self.layout.meta_dir / "manifest.json"
        manifest.save(manifest_path)
        return manifest_path
    
    def finalize(
        self,
        manifest: RunManifest,
        episode_count: int,
        expected_steps: int,
        duration_secs: float,
    ) -> dict[str, Any]:
        """Finalize the dataset.
        
        Args:
            manifest: RunManifest to seal.
            episode_count: Number of episodes.
            expected_steps: Expected total steps.
            duration_secs: Build duration.
        
        Returns:
            Dict with finalization results.
        
        Raises:
            ValueError: If verification fails.
        """
        # Verify offsets
        errors = self.verify_offsets(episode_count, expected_steps)
        if errors:
            manifest.fail("; ".join(errors))
            manifest.save(self.layout.meta_dir / "manifest.json")
            raise ValueError(f"Finalization failed: {errors}")
        
        # Compute and write checksums
        checksums = self.compute_checksums()
        checksums_path = self.write_checksums(checksums)
        
        # Seal manifest
        manifest_path = self.seal_manifest(manifest, episode_count, duration_secs)
        
        return {
            "status": "completed",
            "episode_count": episode_count,
            "checksums_path": str(checksums_path),
            "manifest_path": str(manifest_path),
            "file_count": len(checksums),
        }
