"""Feature statistics schema for normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FeatureStats:
    """Statistics for a single feature.

    Attributes:
        mean: Mean values per dimension.
        std: Standard deviation per dimension.
        min: Minimum values per dimension.
        max: Maximum values per dimension.
        count: Number of samples used to compute stats.
    """

    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)
    min: list[float] = field(default_factory=list)
    max: list[float] = field(default_factory=list)
    count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureStats":
        """Create from dictionary."""
        return cls(
            mean=data.get("mean", []),
            std=data.get("std", []),
            min=data.get("min", []),
            max=data.get("max", []),
            count=data.get("count", 0),
        )

    def normalize(self, value: np.ndarray) -> np.ndarray:
        """Normalize value using stored stats."""
        mean = np.array(self.mean)
        std = np.array(self.std)
        # Avoid division by zero
        std = np.where(std > 1e-8, std, 1.0)
        return (value - mean) / std

    def denormalize(self, value: np.ndarray) -> np.ndarray:
        """Denormalize value using stored stats."""
        mean = np.array(self.mean)
        std = np.array(self.std)
        return value * std + mean


@dataclass
class DatasetStats:
    """Statistics for all features in a dataset.

    Compatible with LeRobot v3 meta/stats.json format.
    """

    features: dict[str, FeatureStats] = field(default_factory=dict)

    def __getitem__(self, key: str) -> FeatureStats | None:
        """Get stats for a feature."""
        return self.features.get(key)

    def __setitem__(self, key: str, stats: FeatureStats) -> None:
        """Set stats for a feature."""
        self.features[key] = stats

    def __contains__(self, key: str) -> bool:
        """Check if feature has stats."""
        return key in self.features

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {key: stats.to_dict() for key, stats in self.features.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetStats":
        """Create from dictionary."""
        features = {key: FeatureStats.from_dict(val) for key, val in data.items()}
        return cls(features=features)

    def to_json(self, path: Path | str) -> None:
        """Write to JSON file (LeRobot v3 format)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path | str) -> "DatasetStats":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


class StatsAccumulator:
    """Online accumulator for computing feature statistics."""

    def __init__(self) -> None:
        """Initialize accumulator."""
        self._count: dict[str, int] = {}
        self._sum: dict[str, np.ndarray] = {}
        self._sum_sq: dict[str, np.ndarray] = {}
        self._min: dict[str, np.ndarray] = {}
        self._max: dict[str, np.ndarray] = {}

    def add(self, key: str, value: np.ndarray) -> None:
        """Add a value to the accumulator."""
        flat = value.flatten().astype(np.float64)

        if key not in self._count:
            self._count[key] = 0
            self._sum[key] = np.zeros_like(flat)
            self._sum_sq[key] = np.zeros_like(flat)
            self._min[key] = np.full_like(flat, np.inf)
            self._max[key] = np.full_like(flat, -np.inf)

        self._count[key] += 1
        self._sum[key] += flat
        self._sum_sq[key] += flat ** 2
        self._min[key] = np.minimum(self._min[key], flat)
        self._max[key] = np.maximum(self._max[key], flat)

    def compute(self) -> DatasetStats:
        """Compute final statistics."""
        stats = DatasetStats()
        for key in self._count:
            n = self._count[key]
            if n == 0:
                continue

            mean = self._sum[key] / n
            variance = (self._sum_sq[key] / n) - (mean ** 2)
            std = np.sqrt(np.maximum(variance, 0))

            stats[key] = FeatureStats(
                mean=mean.tolist(),
                std=std.tolist(),
                min=self._min[key].tolist(),
                max=self._max[key].tolist(),
                count=n,
            )
        return stats
