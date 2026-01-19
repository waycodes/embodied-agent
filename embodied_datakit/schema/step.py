"""Canonical Step dataclass aligned with RLDS semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Step:
    """Canonical step representation aligned with RLDS semantics.

    RLDS Invariants:
    - is_first: True exactly once at step 0
    - is_last: True exactly once at the final step
    - After is_last=True, action/reward/discount are semantically invalid

    Attributes:
        is_first: True at the first step of the episode only.
        is_last: True at the last step of the episode only.
        is_terminal: True if episode ended due to terminal state (vs truncation).
        observation: Flattened observation dict with dotted keys.
        action: Action vector (None after is_last).
        reward: Reward signal (None after is_last).
        discount: Discount factor (None after is_last).
        timestamp: Seconds since episode start.
        step_metadata: Additional step-level metadata.
    """

    is_first: bool
    is_last: bool
    observation: dict[str, np.ndarray | str | bytes]
    timestamp: float = 0.0
    is_terminal: bool = False
    action: np.ndarray | None = None
    reward: float | None = None
    discount: float | None = None
    step_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate step invariants."""
        # Ensure observation is a dict
        if not isinstance(self.observation, dict):
            raise ValueError("observation must be a dict")

    @property
    def has_valid_action(self) -> bool:
        """Check if action is semantically valid (not after is_last)."""
        return not self.is_last and self.action is not None

    def get_image(self, camera: str) -> np.ndarray | None:
        """Get image observation for a camera."""
        key = f"observation.images.{camera}"
        if key in self.observation:
            value = self.observation[key]
            if isinstance(value, np.ndarray):
                return value
        return None

    def get_state(self) -> np.ndarray | None:
        """Get proprioceptive state observation."""
        if "observation.state" in self.observation:
            value = self.observation["observation.state"]
            if isinstance(value, np.ndarray):
                return value
        return None

    def get_language(self) -> str | None:
        """Get language/task instruction."""
        if "observation.language" in self.observation:
            value = self.observation["observation.language"]
            if isinstance(value, str):
                return value
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert step to dictionary for serialization."""
        result: dict[str, Any] = {
            "is_first": self.is_first,
            "is_last": self.is_last,
            "is_terminal": self.is_terminal,
            "timestamp": self.timestamp,
            "observation": {},
            "step_metadata": self.step_metadata.copy(),
        }

        # Serialize observations
        for key, value in self.observation.items():
            if isinstance(value, np.ndarray):
                result["observation"][key] = value.tolist()
            else:
                result["observation"][key] = value

        # Serialize action
        if self.action is not None:
            result["action"] = self.action.tolist()
        else:
            result["action"] = None

        result["reward"] = self.reward
        result["discount"] = self.discount

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Step":
        """Create step from dictionary."""
        observation = {}
        for key, value in data.get("observation", {}).items():
            if isinstance(value, list):
                observation[key] = np.array(value)
            else:
                observation[key] = value

        action = None
        if data.get("action") is not None:
            action = np.array(data["action"])

        return cls(
            is_first=data["is_first"],
            is_last=data["is_last"],
            is_terminal=data.get("is_terminal", False),
            timestamp=data.get("timestamp", 0.0),
            observation=observation,
            action=action,
            reward=data.get("reward"),
            discount=data.get("discount"),
            step_metadata=data.get("step_metadata", {}),
        )
