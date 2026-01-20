"""Deterministic split assignment and mixture spec generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


def deterministic_split(
    episode_id: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Literal["train", "val", "test"]:
    """Assign episode to split deterministically via hash.

    Args:
        episode_id: Unique episode identifier.
        train_ratio: Fraction for train split.
        val_ratio: Fraction for validation split.
        seed: Seed for determinism.

    Returns:
        Split name: 'train', 'val', or 'test'.
    """
    h = hashlib.md5(f"{seed}:{episode_id}".encode()).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF

    if bucket < train_ratio:
        return "train"
    elif bucket < train_ratio + val_ratio:
        return "val"
    return "test"


@dataclass
class DatasetWeight:
    """Weight specification for a dataset in mixture."""

    name: str
    weight: float = 1.0
    filter: dict = field(default_factory=dict)


@dataclass
class MixtureSpec:
    """Specification for dataset mixture."""

    name: str
    datasets: list[DatasetWeight]
    seed: int = 42

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "datasets": [
                {"name": d.name, "weight": d.weight, "filter": d.filter}
                for d in self.datasets
            ],
        }

    def save_yaml(self, path: Path) -> None:
        """Save mixture spec to YAML file."""
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: Path) -> "MixtureSpec":
        """Load mixture spec from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            name=data["name"],
            seed=data.get("seed", 42),
            datasets=[
                DatasetWeight(
                    name=d["name"],
                    weight=d.get("weight", 1.0),
                    filter=d.get("filter", {}),
                )
                for d in data["datasets"]
            ],
        )


def generate_mixture_spec(
    datasets: list[tuple[str, float]],
    name: str = "mixture",
    seed: int = 42,
) -> MixtureSpec:
    """Generate mixture spec from dataset names and weights.

    Args:
        datasets: List of (dataset_name, weight) tuples.
        name: Mixture name.
        seed: Random seed.

    Returns:
        MixtureSpec instance.
    """
    return MixtureSpec(
        name=name,
        seed=seed,
        datasets=[DatasetWeight(name=n, weight=w) for n, w in datasets],
    )
