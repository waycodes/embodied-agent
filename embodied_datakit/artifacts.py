"""Artifact layout conventions for EmbodiedDataKit outputs."""

from __future__ import annotations

from pathlib import Path

# Standard subdirectory names
DIR_DATA = "data"           # Parquet step data
DIR_VIDEOS = "videos"       # MP4 video files
DIR_META = "meta"           # Metadata files (info.json, tasks.jsonl, episodes.parquet)
DIR_REPORTS = "reports"     # Validation reports
DIR_LOGS = "logs"           # Run logs


class ArtifactLayout:
    """Standard artifact layout for compiled datasets.
    
    Structure:
        {output_dir}/
            meta/
                info.json           # Dataset metadata
                tasks.jsonl         # Task catalog
                episodes.parquet    # Episode index
                stats.json          # Dataset statistics
            data/
                chunk-000/
                    episode_000000.parquet
                    ...
            videos/
                chunk-000/
                    episode_000000_{camera}.mp4
                    ...
            reports/
                validation.json     # Validation report
            logs/
                compile.jsonl       # Compilation log
    """
    
    def __init__(self, output_dir: Path | str) -> None:
        """Initialize artifact layout.
        
        Args:
            output_dir: Root output directory.
        """
        self.root = Path(output_dir)
    
    @property
    def meta_dir(self) -> Path:
        """Metadata directory."""
        return self.root / DIR_META
    
    @property
    def data_dir(self) -> Path:
        """Data directory for Parquet files."""
        return self.root / DIR_DATA
    
    @property
    def videos_dir(self) -> Path:
        """Videos directory for MP4 files."""
        return self.root / DIR_VIDEOS
    
    @property
    def reports_dir(self) -> Path:
        """Reports directory."""
        return self.root / DIR_REPORTS
    
    @property
    def logs_dir(self) -> Path:
        """Logs directory."""
        return self.root / DIR_LOGS
    
    # Standard file paths
    @property
    def info_path(self) -> Path:
        """Dataset info JSON."""
        return self.meta_dir / "info.json"
    
    @property
    def tasks_path(self) -> Path:
        """Tasks JSONL file."""
        return self.meta_dir / "tasks.jsonl"
    
    @property
    def episodes_index_path(self) -> Path:
        """Episodes index Parquet."""
        return self.meta_dir / "episodes.parquet"
    
    @property
    def stats_path(self) -> Path:
        """Dataset statistics JSON."""
        return self.meta_dir / "stats.json"
    
    @property
    def validation_report_path(self) -> Path:
        """Validation report JSON."""
        return self.reports_dir / "validation.json"
    
    @property
    def compile_log_path(self) -> Path:
        """Compilation log JSONL."""
        return self.logs_dir / "compile.jsonl"
    
    def chunk_data_dir(self, chunk_idx: int) -> Path:
        """Get data directory for a chunk."""
        return self.data_dir / f"chunk-{chunk_idx:03d}"
    
    def chunk_videos_dir(self, chunk_idx: int) -> Path:
        """Get videos directory for a chunk."""
        return self.videos_dir / f"chunk-{chunk_idx:03d}"
    
    def episode_parquet_path(self, chunk_idx: int, episode_idx: int) -> Path:
        """Get Parquet path for an episode."""
        return self.chunk_data_dir(chunk_idx) / f"episode_{episode_idx:06d}.parquet"
    
    def episode_video_path(self, chunk_idx: int, episode_idx: int, camera: str) -> Path:
        """Get video path for an episode camera."""
        return self.chunk_videos_dir(chunk_idx) / f"episode_{episode_idx:06d}_{camera}.mp4"
    
    def create_dirs(self) -> None:
        """Create all standard directories."""
        for d in [self.meta_dir, self.data_dir, self.videos_dir, self.reports_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)
