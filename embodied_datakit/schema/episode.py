"""Canonical Episode dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from embodied_datakit.schema.step import Step


@dataclass
class Episode:
    """Canonical episode representation.

    Attributes:
        episode_id: Globally unique episode identifier.
        dataset_id: Source dataset identifier.
        steps: Ordered sequence of steps.
        task_id: Integer ID for task (from TaskCatalog).
        task_text: Natural language task description.
        invalid: RLDS invalid flag for episodes to skip in training.
        episode_metadata: Additional episode-level metadata.
    """

    episode_id: str
    dataset_id: str
    steps: list[Step]
    task_id: int = 0
    task_text: str = ""
    invalid: bool = False
    episode_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate episode invariants."""
        if not self.steps:
            return  # Empty episode allowed for validation purposes

        # Skip validation if invalid flag is set (for testing validators)
        if self.invalid:
            return

        # Validate RLDS invariants
        if not self.steps[0].is_first:
            raise ValueError("First step must have is_first=True")
        if not self.steps[-1].is_last:
            raise ValueError("Last step must have is_last=True")

    @property
    def num_steps(self) -> int:
        """Get number of steps."""
        return len(self.steps)

    @property
    def duration(self) -> float:
        """Get episode duration in seconds."""
        if not self.steps:
            return 0.0
        return self.steps[-1].timestamp - self.steps[0].timestamp

    @property
    def is_terminal(self) -> bool:
        """Check if episode ended in terminal state."""
        if not self.steps:
            return False
        return self.steps[-1].is_terminal

    def iter_steps(self) -> Iterator[Step]:
        """Iterate over steps."""
        return iter(self.steps)

    def get_step(self, index: int) -> Step:
        """Get step by index."""
        return self.steps[index]

    def get_observations(self, key: str) -> list[Any]:
        """Get observation values for a key across all steps."""
        return [step.observation.get(key) for step in self.steps]

    def get_actions(self) -> list[Any]:
        """Get all actions (excluding last step where action is invalid)."""
        return [step.action for step in self.steps[:-1] if step.action is not None]

    def get_camera_names(self) -> set[str]:
        """Get set of camera names present in observations."""
        cameras = set()
        for step in self.steps:
            for key in step.observation:
                if key.startswith("observation.images."):
                    camera = key.split(".")[-1]
                    cameras.add(camera)
        return cameras

    def to_dict(self) -> dict[str, Any]:
        """Convert episode to dictionary for serialization."""
        return {
            "episode_id": self.episode_id,
            "dataset_id": self.dataset_id,
            "task_id": self.task_id,
            "task_text": self.task_text,
            "invalid": self.invalid,
            "num_steps": self.num_steps,
            "duration": self.duration,
            "episode_metadata": self.episode_metadata.copy(),
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Episode":
        """Create episode from dictionary."""
        steps = [Step.from_dict(s) for s in data.get("steps", [])]
        return cls(
            episode_id=data["episode_id"],
            dataset_id=data["dataset_id"],
            steps=steps,
            task_id=data.get("task_id", 0),
            task_text=data.get("task_text", ""),
            invalid=data.get("invalid", False),
            episode_metadata=data.get("episode_metadata", {}),
        )

    def validate_structure(self) -> list[str]:
        """Validate RLDS structural invariants, returning list of issues."""
        issues = []

        if not self.steps:
            issues.append("Episode has no steps")
            return issues

        # Check is_first
        is_first_count = sum(1 for s in self.steps if s.is_first)
        if is_first_count != 1:
            issues.append(f"is_first=True appears {is_first_count} times, expected 1")
        if not self.steps[0].is_first:
            issues.append("is_first=True not at step 0")

        # Check is_last
        is_last_count = sum(1 for s in self.steps if s.is_last)
        if is_last_count != 1:
            issues.append(f"is_last=True appears {is_last_count} times, expected 1")
        if not self.steps[-1].is_last:
            issues.append("is_last=True not at final step")

        # Check consistent observation keys
        if self.steps:
            first_keys = set(self.steps[0].observation.keys())
            for i, step in enumerate(self.steps[1:], 1):
                step_keys = set(step.observation.keys())
                if step_keys != first_keys:
                    missing = first_keys - step_keys
                    extra = step_keys - first_keys
                    if missing:
                        issues.append(f"Step {i} missing keys: {missing}")
                    if extra:
                        issues.append(f"Step {i} has extra keys: {extra}")

        return issues
