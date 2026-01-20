"""Canonical action representation types and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    """Canonical action space types."""
    
    EE_DELTA_7 = "ee_delta_7"      # Delta end-effector (dx,dy,dz,drx,dry,drz,gripper)
    EE_ABS_7 = "ee_abs_7"          # Absolute end-effector pose
    EE_VELOCITY_7 = "ee_velocity_7"  # Velocity end-effector
    JOINT_DELTA = "joint_delta"    # Delta joint angles (variable dim)
    JOINT_ABS = "joint_abs"        # Absolute joint angles (variable dim)
    JOINT_VELOCITY = "joint_velocity"  # Joint velocities
    CUSTOM = "custom"              # Dataset-specific


# Standard dimensions for action types
ACTION_DIMS: dict[ActionType, int | None] = {
    ActionType.EE_DELTA_7: 7,
    ActionType.EE_ABS_7: 7,
    ActionType.EE_VELOCITY_7: 7,
    ActionType.JOINT_DELTA: None,  # Variable
    ActionType.JOINT_ABS: None,
    ActionType.JOINT_VELOCITY: None,
    ActionType.CUSTOM: None,
}


@dataclass
class ActionMetadata:
    """Metadata describing action space semantics.
    
    Attributes:
        action_type: Canonical action type enum.
        dim: Action dimension.
        coordinate_frame: Reference frame (e.g., "base", "ee", "world").
        rotation_repr: Rotation representation (e.g., "euler_xyz", "axis_angle", "quat").
        gripper_range: (min, max) gripper values.
        units: Physical units (e.g., "meters", "radians").
        joint_names: Names of joints for joint-space actions.
    """
    
    action_type: ActionType = ActionType.CUSTOM
    dim: int = 7
    coordinate_frame: str = "base"
    rotation_repr: str = "euler_xyz"
    gripper_range: tuple[float, float] = (0.0, 1.0)
    units: str = "meters"
    joint_names: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_type": self.action_type.value,
            "dim": self.dim,
            "coordinate_frame": self.coordinate_frame,
            "rotation_repr": self.rotation_repr,
            "gripper_range": list(self.gripper_range),
            "units": self.units,
            "joint_names": self.joint_names,
            "extra": self.extra,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionMetadata":
        """Create from dictionary."""
        action_type = ActionType(data.get("action_type", "custom"))
        gripper_range = tuple(data.get("gripper_range", [0.0, 1.0]))
        return cls(
            action_type=action_type,
            dim=data.get("dim", 7),
            coordinate_frame=data.get("coordinate_frame", "base"),
            rotation_repr=data.get("rotation_repr", "euler_xyz"),
            gripper_range=gripper_range,
            units=data.get("units", "meters"),
            joint_names=data.get("joint_names", []),
            extra=data.get("extra", {}),
        )
    
    def validate(self) -> list[str]:
        """Validate action metadata, return list of issues."""
        issues = []
        expected_dim = ACTION_DIMS.get(self.action_type)
        if expected_dim is not None and self.dim != expected_dim:
            issues.append(f"Dimension {self.dim} doesn't match {self.action_type.value} (expected {expected_dim})")
        if self.dim <= 0:
            issues.append(f"Invalid dimension: {self.dim}")
        return issues
