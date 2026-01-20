"""Video encoding pipeline for MP4 output."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class VideoOffset:
    """Offset information for an episode in a video file."""
    
    video_file: str
    start_frame: int
    num_frames: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "video_file": self.video_file,
            "start_frame": self.start_frame,
            "num_frames": self.num_frames,
        }


@dataclass
class VideoEncoder:
    """Encode frames to MP4 using ffmpeg.
    
    Attributes:
        output_path: Path to output MP4 file.
        fps: Frames per second.
        crf: Constant rate factor (quality, lower = better).
        preset: Encoding preset (ultrafast to veryslow).
        pix_fmt: Pixel format.
    """
    
    output_path: Path
    fps: float = 10.0
    crf: int = 23
    preset: str = "medium"
    pix_fmt: str = "yuv420p"
    _process: subprocess.Popen | None = field(default=None, repr=False)
    _frame_count: int = 0
    _width: int = 0
    _height: int = 0
    
    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def start(self, width: int, height: int) -> None:
        """Start the encoder process.
        
        Args:
            width: Frame width.
            height: Frame height.
        """
        self._width = width
        self._height = height
        self._frame_count = 0
        
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(self.fps),
            "-i", "-",  # Read from stdin
            "-c:v", "libx264",
            "-crf", str(self.crf),
            "-preset", self.preset,
            "-pix_fmt", self.pix_fmt,
            str(self.output_path),
        ]
        
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    
    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single frame.
        
        Args:
            frame: RGB frame as uint8 array (H, W, 3).
        """
        if self._process is None:
            raise RuntimeError("Encoder not started")
        
        # Ensure correct format
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        
        self._process.stdin.write(frame.tobytes())
        self._frame_count += 1
    
    def finish(self) -> int:
        """Finish encoding and return frame count."""
        if self._process is not None:
            self._process.stdin.close()
            self._process.wait()
            self._process = None
        return self._frame_count
    
    @property
    def frame_count(self) -> int:
        """Number of frames written."""
        return self._frame_count


class VideoShardWriter:
    """Write video shards with episode offset tracking."""
    
    def __init__(
        self,
        output_dir: Path | str,
        camera_name: str,
        fps: float = 10.0,
        crf: int = 23,
        max_frames_per_shard: int = 10000,
    ) -> None:
        """Initialize video shard writer.
        
        Args:
            output_dir: Directory for video files.
            camera_name: Camera name for file naming.
            fps: Frames per second.
            crf: Quality factor.
            max_frames_per_shard: Max frames before starting new shard.
        """
        self.output_dir = Path(output_dir)
        self.camera_name = camera_name
        self.fps = fps
        self.crf = crf
        self.max_frames_per_shard = max_frames_per_shard
        
        self._shard_idx = 0
        self._encoder: VideoEncoder | None = None
        self._shard_frame_count = 0
        self._episode_offsets: dict[str, VideoOffset] = {}
    
    def _get_shard_path(self) -> Path:
        """Get path for current shard."""
        return self.output_dir / f"{self.camera_name}_{self._shard_idx:03d}.mp4"
    
    def _start_new_shard(self, width: int, height: int) -> None:
        """Start a new video shard."""
        if self._encoder is not None:
            self._encoder.finish()
        
        self._encoder = VideoEncoder(
            output_path=self._get_shard_path(),
            fps=self.fps,
            crf=self.crf,
        )
        self._encoder.start(width, height)
        self._shard_frame_count = 0
    
    def write_episode_frames(
        self, episode_id: str, frames: list[np.ndarray]
    ) -> VideoOffset:
        """Write frames for an episode.
        
        Args:
            episode_id: Episode identifier.
            frames: List of RGB frames.
        
        Returns:
            VideoOffset for this episode.
        """
        if not frames:
            return VideoOffset(video_file="", start_frame=0, num_frames=0)
        
        height, width = frames[0].shape[:2]
        
        # Check if we need a new shard
        if (
            self._encoder is None
            or self._shard_frame_count + len(frames) > self.max_frames_per_shard
        ):
            if self._encoder is not None:
                self._encoder.finish()
                self._shard_idx += 1
            self._start_new_shard(width, height)
        
        # Record offset
        start_frame = self._shard_frame_count
        video_file = str(self._get_shard_path().name)
        
        # Write frames
        for frame in frames:
            self._encoder.write_frame(frame)
            self._shard_frame_count += 1
        
        offset = VideoOffset(
            video_file=video_file,
            start_frame=start_frame,
            num_frames=len(frames),
        )
        self._episode_offsets[episode_id] = offset
        return offset
    
    def finish(self) -> dict[str, VideoOffset]:
        """Finish writing and return all offsets."""
        if self._encoder is not None:
            self._encoder.finish()
            self._encoder = None
        return self._episode_offsets
    
    def get_offsets_json(self) -> str:
        """Get offsets as JSON string."""
        return json.dumps({
            ep_id: off.to_dict() for ep_id, off in self._episode_offsets.items()
        })
