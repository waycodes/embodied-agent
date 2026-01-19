"""Base transform interface for episode canonicalization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.spec import DatasetSpec


@runtime_checkable
class Transform(Protocol):
    """Protocol for episode transforms.

    Transforms modify episodes for canonicalization.
    """

    @property
    def name(self) -> str:
        """Transform name for pipeline tracking."""
        ...

    def transform_episode(self, episode: Episode, spec: DatasetSpec) -> Episode:
        """Transform an episode.

        Args:
            episode: Input episode.
            spec: Dataset specification.

        Returns:
            Transformed episode (may be same object, modified in place).
        """
        ...


class BaseTransform(ABC):
    """Abstract base class for transforms."""

    @property
    def name(self) -> str:
        """Transform name (defaults to class name)."""
        return self.__class__.__name__

    @abstractmethod
    def transform_episode(self, episode: Episode, spec: DatasetSpec) -> Episode:
        """Transform an episode."""
        pass

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}()"


class TransformChain:
    """Chain of transforms applied sequentially."""

    def __init__(self, transforms: list[Transform] | None = None) -> None:
        """Initialize transform chain.

        Args:
            transforms: List of transforms to apply in order.
        """
        self.transforms: list[Transform] = transforms or []

    def add(self, transform: Transform) -> "TransformChain":
        """Add a transform to the chain.

        Args:
            transform: Transform to add.

        Returns:
            Self for method chaining.
        """
        self.transforms.append(transform)
        return self

    def transform_episode(self, episode: Episode, spec: DatasetSpec) -> Episode:
        """Apply all transforms to an episode.

        Args:
            episode: Input episode.
            spec: Dataset specification.

        Returns:
            Transformed episode.
        """
        for transform in self.transforms:
            episode = transform.transform_episode(episode, spec)
        return episode

    @property
    def names(self) -> list[str]:
        """Get names of all transforms in chain."""
        return [t.name for t in self.transforms]

    def __len__(self) -> int:
        """Number of transforms in chain."""
        return len(self.transforms)

    def __repr__(self) -> str:
        """String representation."""
        return f"TransformChain({self.names})"


class IdentityTransform(BaseTransform):
    """Transform that returns episode unchanged."""

    def transform_episode(self, episode: Episode, spec: DatasetSpec) -> Episode:
        """Return episode unchanged."""
        return episode
