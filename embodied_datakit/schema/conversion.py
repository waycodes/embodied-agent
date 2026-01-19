"""Tensor conversion utilities for serialization."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_numpy(value: Any) -> np.ndarray | Any:
    """Convert value to numpy array if possible.

    Handles:
    - numpy arrays (passthrough)
    - TensorFlow tensors
    - PyTorch tensors
    - Lists/tuples
    - Scalars
    """
    # Already numpy
    if isinstance(value, np.ndarray):
        return value

    # TensorFlow tensor
    if hasattr(value, "numpy") and callable(value.numpy):
        try:
            return value.numpy()
        except Exception:
            pass

    # PyTorch tensor
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            return value.detach().cpu().numpy()
        except Exception:
            pass

    # Lists/tuples
    if isinstance(value, (list, tuple)):
        return np.array(value)

    # Scalars
    if isinstance(value, (int, float, bool)):
        return np.array(value)

    # Bytes/string - return as-is
    if isinstance(value, (bytes, str)):
        return value

    # Unknown type - try conversion
    try:
        return np.array(value)
    except Exception:
        return value


def ensure_serializable(value: Any) -> np.ndarray | str | bytes | int | float | bool | None:
    """Ensure value is serializable (numpy, scalar, or string).

    Raises:
        ValueError: If value cannot be made serializable.
    """
    # None
    if value is None:
        return None

    # Strings and bytes
    if isinstance(value, (str, bytes)):
        return value

    # Python scalars
    if isinstance(value, (int, float, bool)):
        return value

    # Convert to numpy
    result = to_numpy(value)

    if isinstance(result, np.ndarray):
        return result
    if isinstance(result, (str, bytes, int, float, bool)):
        return result

    raise ValueError(f"Cannot serialize value of type {type(value)}")


def flatten_observation(obs: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested observation dict to dotted keys.

    Example:
        {"images": {"front": array}} -> {"observation.images.front": array}
    """
    result = {}

    for key, value in obs.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            # Recursively flatten
            nested = flatten_observation(value, full_key)
            result.update(nested)
        else:
            # Leaf value
            result[full_key] = ensure_serializable(value)

    return result


def unflatten_observation(flat: dict[str, Any]) -> dict[str, Any]:
    """Unflatten dotted keys back to nested dict.

    Example:
        {"observation.images.front": array} -> {"observation": {"images": {"front": array}}}
    """
    result: dict[str, Any] = {}

    for key, value in flat.items():
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    return result


def decode_bytes_string(value: bytes | str) -> str:
    """Decode bytes to string if needed."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def normalize_dtype(dtype: Any) -> str:
    """Normalize dtype to string representation."""
    if isinstance(dtype, str):
        return dtype

    # Numpy dtype
    if hasattr(dtype, "name"):
        return dtype.name

    # TensorFlow dtype
    if hasattr(dtype, "as_numpy_dtype"):
        return dtype.as_numpy_dtype().name

    return str(dtype)
