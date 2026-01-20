"""RLBench evaluation protocol configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class RLBenchProtocol:
    """Evaluation protocol configuration for RLBench."""

    tasks: list[str] = field(default_factory=lambda: ["reach_target"])
    episodes_per_task: int = 25
    seeds: list[int] = field(default_factory=lambda: list(range(5)))
    max_episode_length: int = 200
    headless: bool = True
    image_size: tuple[int, int] = (128, 128)
    cameras: list[str] = field(default_factory=lambda: ["front", "wrist"])

    def to_dict(self) -> dict:
        return {
            "tasks": self.tasks,
            "episodes_per_task": self.episodes_per_task,
            "seeds": self.seeds,
            "max_episode_length": self.max_episode_length,
            "headless": self.headless,
            "image_size": list(self.image_size),
            "cameras": self.cameras,
        }

    def save_yaml(self, path: Path) -> None:
        """Save protocol to YAML."""
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: Path) -> "RLBenchProtocol":
        """Load protocol from YAML."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            tasks=data.get("tasks", ["reach_target"]),
            episodes_per_task=data.get("episodes_per_task", 25),
            seeds=data.get("seeds", list(range(5))),
            max_episode_length=data.get("max_episode_length", 200),
            headless=data.get("headless", True),
            image_size=tuple(data.get("image_size", [128, 128])),
            cameras=data.get("cameras", ["front", "wrist"]),
        )


# Default protocol for standard evaluation
DEFAULT_PROTOCOL = RLBenchProtocol(
    tasks=[
        "reach_target",
        "push_button",
        "pick_and_lift",
        "pick_up_cup",
        "put_groceries_in_cupboard",
    ],
    episodes_per_task=25,
    seeds=[0, 1, 2, 3, 4],
    max_episode_length=200,
    headless=True,
)
