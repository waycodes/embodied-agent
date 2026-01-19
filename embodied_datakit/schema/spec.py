"""DatasetSpec and FeatureSpec for schema and modality registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from embodied_datakit.schema.tasks import TaskCatalog

ActionSpaceType = Literal[
    "ee_delta_7",      # Delta end-effector (x,y,z,r,p,y,g)
    "ee_abs_7",        # Absolute end-effector
    "ee_velocity_7",   # Velocity end-effector
    "joint_delta_n",   # Delta joint angles
    "joint_abs_n",     # Absolute joint angles
    "custom",          # Dataset-specific
]


@dataclass
class FeatureSpec:
    """Specification for a single feature.

    Attributes:
        dtype: Data type string (e.g., "float32", "uint8").
        shape: Shape tuple (e.g., (256, 256, 3)).
        description: Human-readable description.
        is_video: True if feature is video-encoded.
    """

    dtype: str
    shape: tuple[int, ...]
    description: str = ""
    is_video: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "description": self.description,
            "is_video": self.is_video,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureSpec":
        """Create from dictionary."""
        return cls(
            dtype=data["dtype"],
            shape=tuple(data["shape"]),
            description=data.get("description", ""),
            is_video=data.get("is_video", False),
        )


@dataclass
class DatasetSpec:
    """Dataset schema and modality registry.

    Attributes:
        dataset_id: Unique dataset identifier.
        dataset_name: Human-readable dataset name.
        observation_schema: Map of observation keys to FeatureSpec.
        action_schema: FeatureSpec for action.
        control_rate_hz: Control rate in Hz.
        action_space_type: Type of action space.
        camera_names: List of camera names.
        canonical_camera: Selected canonical camera.
        task_catalog: Task text to ID mapping.
        source_uri: Original data location.
        build_id: Build identifier for provenance.
        transform_pipeline: List of applied transforms.
        edk_schema_version: Schema version.
    """

    dataset_id: str
    dataset_name: str
    observation_schema: dict[str, FeatureSpec] = field(default_factory=dict)
    action_schema: FeatureSpec | None = None
    control_rate_hz: float = 10.0
    action_space_type: ActionSpaceType = "custom"
    camera_names: list[str] = field(default_factory=list)
    canonical_camera: str | None = None
    task_catalog: TaskCatalog = field(default_factory=TaskCatalog)
    source_uri: str = ""
    build_id: str = ""
    transform_pipeline: list[str] = field(default_factory=list)
    edk_schema_version: str = "1.0.0"
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_cameras(self) -> int:
        """Get number of cameras."""
        return len(self.camera_names)

    @property
    def has_video(self) -> bool:
        """Check if any features are video-encoded."""
        return any(spec.is_video for spec in self.observation_schema.values())

    def get_image_features(self) -> dict[str, FeatureSpec]:
        """Get all image observation features."""
        return {
            k: v for k, v in self.observation_schema.items()
            if k.startswith("observation.images.")
        }

    def get_state_feature(self) -> FeatureSpec | None:
        """Get proprioceptive state feature."""
        return self.observation_schema.get("observation.state")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "observation_schema": {
                k: v.to_dict() for k, v in self.observation_schema.items()
            },
            "action_schema": self.action_schema.to_dict() if self.action_schema else None,
            "control_rate_hz": self.control_rate_hz,
            "action_space_type": self.action_space_type,
            "camera_names": self.camera_names,
            "canonical_camera": self.canonical_camera,
            "task_catalog": self.task_catalog.to_dict(),
            "source_uri": self.source_uri,
            "build_id": self.build_id,
            "transform_pipeline": self.transform_pipeline,
            "edk_schema_version": self.edk_schema_version,
            "extra_metadata": self.extra_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetSpec":
        """Create from dictionary."""
        observation_schema = {
            k: FeatureSpec.from_dict(v)
            for k, v in data.get("observation_schema", {}).items()
        }

        action_schema = None
        if data.get("action_schema"):
            action_schema = FeatureSpec.from_dict(data["action_schema"])

        task_catalog = TaskCatalog()
        if data.get("task_catalog"):
            task_catalog = TaskCatalog.from_dict(data["task_catalog"])

        return cls(
            dataset_id=data["dataset_id"],
            dataset_name=data["dataset_name"],
            observation_schema=observation_schema,
            action_schema=action_schema,
            control_rate_hz=data.get("control_rate_hz", 10.0),
            action_space_type=data.get("action_space_type", "custom"),
            camera_names=data.get("camera_names", []),
            canonical_camera=data.get("canonical_camera"),
            task_catalog=task_catalog,
            source_uri=data.get("source_uri", ""),
            build_id=data.get("build_id", ""),
            transform_pipeline=data.get("transform_pipeline", []),
            edk_schema_version=data.get("edk_schema_version", "1.0.0"),
            extra_metadata=data.get("extra_metadata", {}),
        )
