"""Configuration system for EmbodiedDataKit."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ValidationConfig:
    """Validation configuration."""

    fail_fast: bool = False
    fail_on_warn: bool = False
    quarantine: bool = False
    strict: bool = False
    min_episode_length: int = 1
    max_episode_length: int = 100000
    action_bounds: tuple[float, float] = (-10.0, 10.0)
    action_sigma_threshold: float = 5.0
    timestamp_gap_factor: float = 2.0
    severity_overrides: dict[str, str] = field(default_factory=dict)


@dataclass
class ShardingConfig:
    """Sharding configuration."""

    episodes_per_parquet_shard: int = 1000
    max_video_frames_per_shard: int = 10000
    video_crf: int = 23
    video_preset: str = "medium"


@dataclass
class TransformConfig:
    """Transform configuration."""

    camera: str | None = None
    camera_fallback_order: list[str] = field(
        default_factory=lambda: ["front", "workspace", "overhead", "wrist"]
    )
    resolution: tuple[int, int] = (256, 256)
    action_mapping: str = "passthrough"
    normalize_actions: bool = False
    flatten_keys: bool = True
    add_episode_ids: bool = True
    mark_invalid: bool = True


@dataclass
class Config:
    """Main configuration for EmbodiedDataKit."""

    # Output settings
    output_dir: Path | None = None

    # Execution settings
    workers: int = 1
    seed: int = 42
    resume: bool = False

    # Sub-configs
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    sharding: ShardingConfig = field(default_factory=ShardingConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)

    # Pipeline-specific overrides by dataset
    dataset_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "Config":
        """Build config from dictionary."""
        validation_data = data.pop("validation", {})
        sharding_data = data.pop("sharding", {})
        transform_data = data.pop("transform", {})

        validation = ValidationConfig(**validation_data) if validation_data else ValidationConfig()
        sharding = ShardingConfig(**sharding_data) if sharding_data else ShardingConfig()
        transform = TransformConfig(**transform_data) if transform_data else TransformConfig()

        # Handle output_dir
        if "output_dir" in data and data["output_dir"]:
            data["output_dir"] = Path(data["output_dir"])

        return cls(
            validation=validation,
            sharding=sharding,
            transform=transform,
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict

        result = asdict(self)
        if result["output_dir"]:
            result["output_dir"] = str(result["output_dir"])
        return result

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def get_dataset_config(self, dataset_name: str) -> "Config":
        """Get config with dataset-specific overrides applied."""
        if dataset_name not in self.dataset_overrides:
            return self

        # Create a copy with overrides
        data = self.to_dict()
        overrides = self.dataset_overrides[dataset_name]

        # Merge overrides
        for key, value in overrides.items():
            if key in data and isinstance(data[key], dict) and isinstance(value, dict):
                data[key].update(value)
            else:
                data[key] = value

        return Config._from_dict(data)


def load_config(path: Path | str | None = None) -> Config:
    """Load configuration from file or return defaults."""
    if path is None:
        return Config()
    return Config.from_yaml(path)
