"""Canonical key naming convention and flattening utilities."""

from __future__ import annotations

from typing import Any

# Separator for nested keys
KEY_SEP = "."

# Standard prefixes
PREFIX_OBSERVATION = "observation"
PREFIX_IMAGES = "observation.images"
PREFIX_DEPTH = "observation.depth"
PREFIX_STATE = "observation.state"
PREFIX_PROPRIO = "observation.proprio"
PREFIX_LANGUAGE = "observation.language"


def flatten_dict(nested: dict[str, Any], sep: str = KEY_SEP) -> dict[str, Any]:
    """Flatten nested dict to dotted keys.
    
    Example:
        {"observation": {"images": {"front": arr}}} -> {"observation.images.front": arr}
    """
    result: dict[str, Any] = {}
    
    def _flatten(obj: Any, prefix: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}{sep}{k}" if prefix else k
                _flatten(v, new_key)
        else:
            result[prefix] = obj
    
    _flatten(nested, "")
    return result


def unflatten_dict(flat: dict[str, Any], sep: str = KEY_SEP) -> dict[str, Any]:
    """Unflatten dotted keys to nested dict.
    
    Example:
        {"observation.images.front": arr} -> {"observation": {"images": {"front": arr}}}
    """
    result: dict[str, Any] = {}
    
    for key, value in flat.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    
    return result


def normalize_key(key: str) -> str:
    """Normalize key to canonical form (lowercase, underscores)."""
    return key.lower().replace("-", "_").replace(" ", "_")


def is_image_key(key: str) -> bool:
    """Check if key is an image observation."""
    return key.startswith(PREFIX_IMAGES + KEY_SEP)


def is_depth_key(key: str) -> bool:
    """Check if key is a depth observation."""
    return key.startswith(PREFIX_DEPTH + KEY_SEP)


def get_camera_name(key: str) -> str | None:
    """Extract camera name from image/depth key."""
    if is_image_key(key):
        return key.split(KEY_SEP)[-1]
    if is_depth_key(key):
        return key.split(KEY_SEP)[-1]
    return None


def make_image_key(camera: str) -> str:
    """Create canonical image key for camera."""
    return f"{PREFIX_IMAGES}{KEY_SEP}{normalize_key(camera)}"


def make_depth_key(camera: str) -> str:
    """Create canonical depth key for camera."""
    return f"{PREFIX_DEPTH}{KEY_SEP}{normalize_key(camera)}"
