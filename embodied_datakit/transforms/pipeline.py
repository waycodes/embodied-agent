"""Transform pipeline configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from embodied_datakit.transforms.action import (
    MapActionSpaceTransform,
    NormalizeActionsTransform,
    PadActionTransform,
)
from embodied_datakit.transforms.base import BaseTransform, TransformChain
from embodied_datakit.transforms.camera import ResizeImagesTransform, SelectCameraTransform
from embodied_datakit.transforms.task import TaskTextTransform

# Registry of available transforms
TRANSFORM_REGISTRY: dict[str, type[BaseTransform]] = {
    "select_camera": SelectCameraTransform,
    "resize_images": ResizeImagesTransform,
    "normalize_actions": NormalizeActionsTransform,
    "pad_action": PadActionTransform,
    "map_action_space": MapActionSpaceTransform,
    "task_text": TaskTextTransform,
}


def build_transform(name: str, params: dict[str, Any] | None = None) -> BaseTransform:
    """Build a transform from name and parameters.
    
    Args:
        name: Transform name from registry.
        params: Optional parameters to pass to constructor.
    
    Returns:
        Instantiated transform.
    
    Raises:
        ValueError: If transform name not found.
    """
    if name not in TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown transform: {name}. Available: {list(TRANSFORM_REGISTRY.keys())}")
    
    cls = TRANSFORM_REGISTRY[name]
    params = params or {}
    return cls(**params)


def load_pipeline_config(path: Path | str) -> TransformChain:
    """Load transform pipeline from YAML config.
    
    Config format:
        transforms:
          - name: select_camera
            params:
              camera: front
          - name: resize_images
            params:
              size: [256, 256]
    
    Args:
        path: Path to YAML config file.
    
    Returns:
        TransformChain with configured transforms.
    """
    path = Path(path)
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    
    return build_pipeline_from_config(config)


def build_pipeline_from_config(config: dict[str, Any]) -> TransformChain:
    """Build transform pipeline from config dict.
    
    Args:
        config: Config dict with 'transforms' key.
    
    Returns:
        TransformChain with configured transforms.
    """
    chain = TransformChain()
    
    transforms_config = config.get("transforms", [])
    for t_config in transforms_config:
        name = t_config.get("name")
        if not name:
            continue
        params = t_config.get("params", {})
        transform = build_transform(name, params)
        chain.add(transform)
    
    return chain


def register_transform(name: str, cls: type[BaseTransform]) -> None:
    """Register a custom transform.
    
    Args:
        name: Name to register under.
        cls: Transform class.
    """
    TRANSFORM_REGISTRY[name] = cls
