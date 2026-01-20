"""Run manifest for reproducibility tracking."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import embodied_datakit


@dataclass
class RunManifest:
    """Manifest for tracking build reproducibility.
    
    Attributes:
        build_id: Unique build identifier (hash of config + timestamp).
        timestamp: Build start time (ISO format).
        edk_version: EmbodiedDataKit version.
        git_commit: Git commit hash (if available).
        git_dirty: Whether working tree had uncommitted changes.
        config_hash: Hash of configuration used.
        config: Full configuration dict.
        source_uri: Input data source.
        output_dir: Output directory path.
        artifacts: List of produced artifact paths.
        episode_count: Number of episodes processed.
        duration_secs: Build duration in seconds.
        status: Build status (running/completed/failed).
        error: Error message if failed.
    """
    
    build_id: str
    timestamp: str
    edk_version: str
    git_commit: str = ""
    git_dirty: bool = False
    config_hash: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    source_uri: str = ""
    output_dir: str = ""
    artifacts: list[str] = field(default_factory=list)
    episode_count: int = 0
    duration_secs: float = 0.0
    status: str = "running"
    error: str = ""
    
    @classmethod
    def create(cls, config: dict[str, Any], source_uri: str, output_dir: Path | str) -> "RunManifest":
        """Create a new run manifest.
        
        Args:
            config: Configuration dict.
            source_uri: Input data source URI.
            output_dir: Output directory.
        
        Returns:
            New RunManifest instance.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        config_hash = _hash_config(config)
        build_id = _generate_build_id(config_hash, timestamp)
        git_commit, git_dirty = _get_git_info()
        
        return cls(
            build_id=build_id,
            timestamp=timestamp,
            edk_version=embodied_datakit.__version__,
            git_commit=git_commit,
            git_dirty=git_dirty,
            config_hash=config_hash,
            config=config,
            source_uri=source_uri,
            output_dir=str(output_dir),
        )
    
    def add_artifact(self, path: Path | str) -> None:
        """Add an artifact path."""
        self.artifacts.append(str(path))
    
    def complete(self, episode_count: int, duration_secs: float) -> None:
        """Mark build as completed."""
        self.episode_count = episode_count
        self.duration_secs = duration_secs
        self.status = "completed"
    
    def fail(self, error: str) -> None:
        """Mark build as failed."""
        self.status = "failed"
        self.error = error
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "build_id": self.build_id,
            "timestamp": self.timestamp,
            "edk_version": self.edk_version,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "config_hash": self.config_hash,
            "config": self.config,
            "source_uri": self.source_uri,
            "output_dir": self.output_dir,
            "artifacts": self.artifacts,
            "episode_count": self.episode_count,
            "duration_secs": self.duration_secs,
            "status": self.status,
            "error": self.error,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunManifest":
        """Create from dictionary."""
        return cls(**data)
    
    def save(self, path: Path | str) -> None:
        """Save manifest to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path | str) -> "RunManifest":
        """Load manifest from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _hash_config(config: dict[str, Any]) -> str:
    """Generate hash of configuration."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def _generate_build_id(config_hash: str, timestamp: str) -> str:
    """Generate unique build ID."""
    combined = f"{config_hash}:{timestamp}"
    return hashlib.sha256(combined.encode()).hexdigest()[:12]


def _get_git_info() -> tuple[str, bool]:
    """Get git commit hash and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(status)
        
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "", False
