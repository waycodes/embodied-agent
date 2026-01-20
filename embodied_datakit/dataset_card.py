"""Dataset card generator for publishing compiled datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


@dataclass
class DatasetCard:
    """Dataset card for HuggingFace Hub or similar platforms."""

    name: str
    description: str = ""
    version: str = "1.0.0"
    license: str = "Proprietary"
    authors: list[str] = field(default_factory=list)
    source_datasets: list[str] = field(default_factory=list)
    num_episodes: int = 0
    num_frames: int = 0
    robot_types: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    cameras: list[str] = field(default_factory=list)
    action_space: str = "continuous"
    action_dim: int = 7
    control_rate_hz: float = 10.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_markdown(self) -> str:
        """Generate markdown dataset card."""
        lines = [
            "---",
            f"license: {self.license}",
            "task_categories:",
            "  - robotics",
            "tags:",
            "  - embodied-ai",
            "  - robot-learning",
            "---",
            "",
            f"# {self.name}",
            "",
            self.description or "A compiled robot trajectory dataset.",
            "",
            "## Dataset Summary",
            "",
            f"- **Episodes**: {self.num_episodes:,}",
            f"- **Frames**: {self.num_frames:,}",
            f"- **Control Rate**: {self.control_rate_hz} Hz",
            f"- **Action Dimension**: {self.action_dim}",
            "",
        ]

        if self.robot_types:
            lines.append("## Robot Types")
            lines.append("")
            for robot in self.robot_types:
                lines.append(f"- {robot}")
            lines.append("")

        if self.tasks:
            lines.append("## Tasks")
            lines.append("")
            for task in self.tasks:
                lines.append(f"- {task}")
            lines.append("")

        if self.cameras:
            lines.append("## Cameras")
            lines.append("")
            for cam in self.cameras:
                lines.append(f"- {cam}")
            lines.append("")

        if self.source_datasets:
            lines.append("## Source Datasets")
            lines.append("")
            for src in self.source_datasets:
                lines.append(f"- {src}")
            lines.append("")

        lines.extend([
            "## Citation",
            "",
            "```bibtex",
            "@software{bharadwaj2026embodieddatakit,",
            "  author = {Bharadwaj, Varun},",
            "  title = {EmbodiedDataKit},",
            "  year = {2026},",
            "  url = {https://github.com/waycodes/embodied-agent}",
            "}",
            "```",
        ])

        return "\n".join(lines)

    def save(self, path: Path) -> None:
        """Save dataset card as README.md."""
        with open(path, "w") as f:
            f.write(self.to_markdown())


def generate_card_from_dataset(dataset_path: Path, name: str) -> DatasetCard:
    """Generate dataset card from compiled dataset.

    Args:
        dataset_path: Path to compiled dataset.
        name: Dataset name.

    Returns:
        DatasetCard with populated metadata.
    """
    card = DatasetCard(name=name)

    # Load info.json if exists
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        card.num_episodes = info.get("total_episodes", 0)
        card.num_frames = info.get("total_frames", 0)
        card.control_rate_hz = info.get("fps", 10.0)

    # Load episodes index for more details
    episodes_path = dataset_path / "meta" / "episodes.parquet"
    if not episodes_path.exists():
        episodes_path = dataset_path / "meta" / "episodes" / "episodes.parquet"

    if episodes_path.exists():
        table = pq.read_table(episodes_path)
        rows = table.to_pydict()

        if "robot_id" in rows:
            card.robot_types = list(set(rows["robot_id"]))
        if "task_text" in rows:
            card.tasks = list(set(t for t in rows["task_text"] if t))[:10]
        if "camera_set" in rows:
            cams = set()
            for cs in rows["camera_set"]:
                if cs:
                    cams.update(cs.split(","))
            card.cameras = list(cams)

    # Load tasks.jsonl for task list
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        tasks = []
        with open(tasks_path) as f:
            for line in f:
                if line.strip():
                    task = json.loads(line)
                    tasks.append(task.get("task", task.get("task_text", "")))
        card.tasks = tasks[:10]

    return card
