"""Base writer interface for dataset output."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, runtime_checkable

from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.spec import DatasetSpec


@runtime_checkable
class Writer(Protocol):
    """Protocol for dataset writers.

    Writers serialize episodes to storage formats.
    """

    def begin(self, spec: DatasetSpec, output_dir: Path) -> None:
        """Begin writing a new dataset.

        Args:
            spec: Dataset specification.
            output_dir: Output directory path.
        """
        ...

    def write_episode(self, episode: Episode) -> None:
        """Write a single episode.

        Args:
            episode: Episode to write.
        """
        ...

    def finalize(self) -> list[Path]:
        """Finalize the dataset and return produced artifacts.

        Returns:
            List of paths to produced files.
        """
        ...


class BaseWriter(ABC):
    """Abstract base class for writers."""

    def __init__(self) -> None:
        """Initialize writer."""
        self.spec: DatasetSpec | None = None
        self.output_dir: Path | None = None
        self._episode_count: int = 0
        self._artifacts: list[Path] = []

    @abstractmethod
    def begin(self, spec: DatasetSpec, output_dir: Path) -> None:
        """Begin writing a new dataset."""
        self.spec = spec
        self.output_dir = output_dir
        self._episode_count = 0
        self._artifacts = []

    @abstractmethod
    def write_episode(self, episode: Episode) -> None:
        """Write a single episode."""
        self._episode_count += 1

    @abstractmethod
    def finalize(self) -> list[Path]:
        """Finalize the dataset and return produced artifacts."""
        return self._artifacts

    @property
    def episode_count(self) -> int:
        """Number of episodes written."""
        return self._episode_count

    def __enter__(self) -> "BaseWriter":
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit - calls finalize if not already called."""
        pass
