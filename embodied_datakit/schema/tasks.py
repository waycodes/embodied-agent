"""TaskCatalog for task text to ID mapping."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TaskCatalog:
    """Catalog mapping task text to integer IDs.

    Provides stable task-to-id mapping for slicing and training.
    Compatible with LeRobot v3 meta/tasks.jsonl format.
    """

    _tasks: dict[str, int] = field(default_factory=dict)
    _reverse: dict[int, str] = field(default_factory=dict)
    _next_id: int = 0

    def __len__(self) -> int:
        """Get number of tasks."""
        return len(self._tasks)

    def __contains__(self, task: str) -> bool:
        """Check if task exists."""
        return task in self._tasks

    def add(self, task: str) -> int:
        """Add a task and return its ID. Returns existing ID if already present."""
        if task in self._tasks:
            return self._tasks[task]

        task_id = self._next_id
        self._tasks[task] = task_id
        self._reverse[task_id] = task
        self._next_id += 1
        return task_id

    def get_id(self, task: str) -> int | None:
        """Get ID for a task, or None if not found."""
        return self._tasks.get(task)

    def get_task(self, task_id: int) -> str | None:
        """Get task text for an ID, or None if not found."""
        return self._reverse.get(task_id)

    def get_or_add(self, task: str) -> int:
        """Get existing ID or add new task."""
        return self.add(task)

    def all_tasks(self) -> list[str]:
        """Get all task texts in ID order."""
        return [self._reverse[i] for i in range(len(self._reverse))]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tasks": self._tasks.copy(),
            "next_id": self._next_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskCatalog":
        """Create from dictionary."""
        catalog = cls()
        catalog._tasks = data.get("tasks", {}).copy()
        catalog._reverse = {v: k for k, v in catalog._tasks.items()}
        catalog._next_id = data.get("next_id", len(catalog._tasks))
        return catalog

    def to_jsonl(self, path: Path | str) -> None:
        """Write to JSONL file (LeRobot v3 format)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for task_id in range(len(self._reverse)):
                entry = {"task_index": task_id, "task": self._reverse[task_id]}
                f.write(json.dumps(entry) + "\n")

    @classmethod
    def from_jsonl(cls, path: Path | str) -> "TaskCatalog":
        """Load from JSONL file (LeRobot v3 format)."""
        catalog = cls()
        path = Path(path)
        with open(path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    task_id = entry["task_index"]
                    task = entry["task"]
                    catalog._tasks[task] = task_id
                    catalog._reverse[task_id] = task
                    catalog._next_id = max(catalog._next_id, task_id + 1)
        return catalog
