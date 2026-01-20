"""Action space transforms for canonicalization."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.spec import DatasetSpec
from embodied_datakit.schema.step import Step
from embodied_datakit.transforms.base import BaseTransform

logger = logging.getLogger(__name__)


# Standard action dimensions for different action spaces
ACTION_SPACE_DIMS = {
    "ee_delta_7": 7,  # dx, dy, dz, drx, dry, drz, gripper
    "ee_abs_7": 7,  # x, y, z, rx, ry, rz, gripper
    "joint_7": 7,  # 7 joint positions
    "joint_6": 6,  # 6 joint positions
    "joint_7_gripper": 8,  # 7 joints + gripper
}


class NormalizeActionsTransform(BaseTransform):
    """Normalize actions to [-1, 1] range using dataset statistics."""

    def __init__(
        self,
        action_bounds: tuple[float, float] | None = None,
        clip: bool = True,
    ) -> None:
        """Initialize action normalization transform.

        Args:
            action_bounds: (min, max) bounds for normalization. If None, uses dataset stats.
            clip: Whether to clip values outside bounds.
        """
        super().__init__("normalize_actions")
        self.action_bounds = action_bounds
        self.clip = clip
        self._action_min: np.ndarray | None = None
        self._action_max: np.ndarray | None = None

    def transform_episode(self, episode: Episode, spec: DatasetSpec) -> Episode:
        """Transform episode by normalizing actions."""
        # Compute bounds from episode if not provided
        if self.action_bounds is None:
            self._compute_bounds(episode)
        else:
            lo, hi = self.action_bounds
            self._action_min = np.array(lo)
            self._action_max = np.array(hi)

        new_steps = []
        for step in episode.steps:
            new_step = self._transform_step(step)
            new_steps.append(new_step)

        return Episode(
            episode_id=episode.episode_id,
            dataset_id=episode.dataset_id,
            steps=new_steps,
            task_id=episode.task_id,
            task_text=episode.task_text,
            split=episode.split,
            provenance=episode.provenance,
            invalid=episode.invalid,
        )

    def _compute_bounds(self, episode: Episode) -> None:
        """Compute action bounds from episode."""
        actions = []
        for step in episode.steps:
            if step.action is not None:
                actions.append(step.action)

        if not actions:
            self._action_min = np.array([-1.0])
            self._action_max = np.array([1.0])
            return

        actions_arr = np.stack(actions)
        self._action_min = actions_arr.min(axis=0)
        self._action_max = actions_arr.max(axis=0)

        # Prevent division by zero
        range_size = self._action_max - self._action_min
        zero_range = range_size < 1e-8
        self._action_min[zero_range] = -1.0
        self._action_max[zero_range] = 1.0

    def _transform_step(self, step: Step) -> Step:
        """Normalize action in step."""
        if step.action is None or self._action_min is None:
            return step

        # Normalize to [-1, 1]
        action_range = self._action_max - self._action_min
        normalized = 2.0 * (step.action - self._action_min) / action_range - 1.0

        if self.clip:
            normalized = np.clip(normalized, -1.0, 1.0)

        return Step(
            is_first=step.is_first,
            is_last=step.is_last,
            is_terminal=step.is_terminal,
            observation=step.observation,
            action=normalized.astype(np.float32),
            reward=step.reward,
            discount=step.discount,
            timestamp=step.timestamp,
            frame_index=step.frame_index,
        )


class PadActionTransform(BaseTransform):
    """Pad or truncate actions to target dimension."""

    def __init__(
        self,
        target_dim: int = 7,
        pad_value: float = 0.0,
    ) -> None:
        """Initialize action padding transform.

        Args:
            target_dim: Target action dimension.
            pad_value: Value to use for padding.
        """
        super().__init__("pad_action")
        self.target_dim = target_dim
        self.pad_value = pad_value

    def transform_episode(self, episode: Episode, spec: DatasetSpec) -> Episode:
        """Transform episode by padding actions."""
        new_steps = []
        for step in episode.steps:
            new_step = self._transform_step(step)
            new_steps.append(new_step)

        return Episode(
            episode_id=episode.episode_id,
            dataset_id=episode.dataset_id,
            steps=new_steps,
            task_id=episode.task_id,
            task_text=episode.task_text,
            split=episode.split,
            provenance=episode.provenance,
            invalid=episode.invalid,
        )

    def _transform_step(self, step: Step) -> Step:
        """Pad or truncate action in step."""
        if step.action is None:
            return step

        current_dim = len(step.action)

        if current_dim == self.target_dim:
            return step
        elif current_dim < self.target_dim:
            # Pad
            padding = np.full(self.target_dim - current_dim, self.pad_value, dtype=np.float32)
            padded = np.concatenate([step.action, padding])
        else:
            # Truncate
            padded = step.action[: self.target_dim]

        return Step(
            is_first=step.is_first,
            is_last=step.is_last,
            is_terminal=step.is_terminal,
            observation=step.observation,
            action=padded.astype(np.float32),
            reward=step.reward,
            discount=step.discount,
            timestamp=step.timestamp,
            frame_index=step.frame_index,
        )


class MapActionSpaceTransform(BaseTransform):
    """Map actions between different action space representations.

    Supports mappings like:
    - joint_7_gripper -> ee_delta_7 (extracts gripper, zeros unused dims)
    - joint_6 -> ee_delta_7 (pads gripper)
    """

    def __init__(
        self,
        source_space: str,
        target_space: str = "ee_delta_7",
        gripper_index: int = -1,
    ) -> None:
        """Initialize action space mapping transform.

        Args:
            source_space: Source action space type.
            target_space: Target action space type.
            gripper_index: Index of gripper in source action (-1 for last).
        """
        super().__init__("map_action_space")
        self.source_space = source_space
        self.target_space = target_space
        self.gripper_index = gripper_index

    def transform_episode(self, episode: Episode, spec: DatasetSpec) -> Episode:
        """Transform episode by mapping action space."""
        new_steps = []
        for step in episode.steps:
            new_step = self._transform_step(step)
            new_steps.append(new_step)

        return Episode(
            episode_id=episode.episode_id,
            dataset_id=episode.dataset_id,
            steps=new_steps,
            task_id=episode.task_id,
            task_text=episode.task_text,
            split=episode.split,
            provenance=episode.provenance,
            invalid=episode.invalid,
        )

    def _transform_step(self, step: Step) -> Step:
        """Map action space in step."""
        if step.action is None:
            return step

        target_dim = ACTION_SPACE_DIMS.get(self.target_space, 7)
        mapped = np.zeros(target_dim, dtype=np.float32)

        source_dim = len(step.action)

        # Copy as many dimensions as possible
        copy_dims = min(source_dim, target_dim)
        mapped[:copy_dims] = step.action[:copy_dims]

        # Handle gripper specially for ee_delta_7
        if self.target_space == "ee_delta_7" and source_dim > target_dim:
            # Gripper is last element of target
            mapped[6] = step.action[self.gripper_index]

        return Step(
            is_first=step.is_first,
            is_last=step.is_last,
            is_terminal=step.is_terminal,
            observation=step.observation,
            action=mapped,
            reward=step.reward,
            discount=step.discount,
            timestamp=step.timestamp,
            frame_index=step.frame_index,
        )
