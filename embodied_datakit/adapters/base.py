"""Base adapter interface for data source ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Protocol, runtime_checkable

from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.spec import DatasetSpec


@runtime_checkable
class Adapter(Protocol):
    """Protocol for dataset adapters.

    Adapters convert source datasets into canonical Episode format.
    """

    def probe(self) -> DatasetSpec:
        """Probe the dataset and return its specification.

        Returns:
            DatasetSpec with schema, modalities, and metadata.
        """
        ...

    def iter_episodes(
        self,
        split: str = "train",
        selector: str | None = None,
    ) -> Iterator[Episode]:
        """Iterate over episodes from the dataset.

        Args:
            split: Dataset split name (e.g., "train", "val", "test").
            selector: Optional slice selector (e.g., "[0:100]", "[40:41]").

        Yields:
            Episode objects in source order.
        """
        ...

    def close(self) -> None:
        """Release any resources held by the adapter."""
        ...


class BaseAdapter(ABC):
    """Abstract base class for adapters with common functionality."""

    def __init__(self, source_uri: str) -> None:
        """Initialize adapter.

        Args:
            source_uri: URI or path to the data source.
        """
        self.source_uri = source_uri
        self._spec: DatasetSpec | None = None

    @abstractmethod
    def probe(self) -> DatasetSpec:
        """Probe the dataset and return its specification."""
        pass

    @abstractmethod
    def iter_episodes(
        self,
        split: str = "train",
        selector: str | None = None,
    ) -> Iterator[Episode]:
        """Iterate over episodes from the dataset."""
        pass

    def close(self) -> None:
        """Release any resources held by the adapter."""
        pass

    def __enter__(self) -> "BaseAdapter":
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()

    def get_spec(self) -> DatasetSpec:
        """Get cached spec or probe if not cached."""
        if self._spec is None:
            self._spec = self.probe()
        return self._spec

    @staticmethod
    def parse_selector(selector: str | None) -> tuple[int | None, int | None]:
        """Parse a slice selector string.

        Args:
            selector: Selector like "[0:100]" or "[40:41]".

        Returns:
            Tuple of (start, end) indices, None for unbounded.
        """
        if not selector:
            return None, None

        selector = selector.strip()
        if selector.startswith("[") and selector.endswith("]"):
            selector = selector[1:-1]

        parts = selector.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid selector format: {selector}")

        start = int(parts[0]) if parts[0] else None
        end = int(parts[1]) if parts[1] else None
        return start, end
