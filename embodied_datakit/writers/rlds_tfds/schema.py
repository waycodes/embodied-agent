"""RLDS export schema builder for TFDS features."""

from __future__ import annotations

from typing import Any

from embodied_datakit.schema.spec import DatasetSpec, FeatureSpec

# TFDS dtype mapping
DTYPE_MAP = {
    "float32": "float32",
    "float64": "float64",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "bool": "bool",
    "string": "string",
}


def build_rlds_schema(spec: DatasetSpec) -> dict[str, Any]:
    """Build RLDS-compatible schema from DatasetSpec.
    
    RLDS schema structure:
        episode: {
            steps: {
                observation: {...},
                action: Tensor,
                reward: Tensor,
                discount: Tensor,
                is_first: Tensor,
                is_last: Tensor,
                is_terminal: Tensor,
            }
        }
    
    Args:
        spec: Dataset specification.
    
    Returns:
        RLDS schema dict compatible with TFDS features.
    """
    # Build observation features
    observation_features = {}
    for key, feat_spec in spec.observation_schema.items():
        # Convert dotted key to nested structure
        parts = key.split(".")
        if len(parts) > 1 and parts[0] == "observation":
            # Remove 'observation.' prefix for RLDS
            nested_key = ".".join(parts[1:])
        else:
            nested_key = key
        
        observation_features[nested_key] = _feature_spec_to_tfds(feat_spec)
    
    # Build action feature
    action_feature = None
    if spec.action_schema:
        action_feature = _feature_spec_to_tfds(spec.action_schema)
    else:
        action_feature = {"dtype": "float32", "shape": (7,)}
    
    # Build step schema
    step_schema = {
        "observation": observation_features,
        "action": action_feature,
        "reward": {"dtype": "float32", "shape": ()},
        "discount": {"dtype": "float32", "shape": ()},
        "is_first": {"dtype": "bool", "shape": ()},
        "is_last": {"dtype": "bool", "shape": ()},
        "is_terminal": {"dtype": "bool", "shape": ()},
    }
    
    # Build episode schema
    episode_schema = {
        "steps": step_schema,
    }
    
    return {
        "episode": episode_schema,
        "metadata": {
            "dataset_name": spec.dataset_name,
            "dataset_id": spec.dataset_id,
            "edk_schema_version": spec.edk_schema_version,
        },
    }


def _feature_spec_to_tfds(feat_spec: FeatureSpec) -> dict[str, Any]:
    """Convert FeatureSpec to TFDS feature dict."""
    dtype = DTYPE_MAP.get(feat_spec.dtype, feat_spec.dtype)
    return {
        "dtype": dtype,
        "shape": feat_spec.shape,
        "description": feat_spec.description,
    }


def build_tfds_features_dict(spec: DatasetSpec) -> dict[str, Any]:
    """Build TFDS FeaturesDict-compatible structure.
    
    This creates a structure that can be used with:
        tfds.features.FeaturesDict(features_dict)
    
    Args:
        spec: Dataset specification.
    
    Returns:
        Dict compatible with TFDS FeaturesDict.
    """
    schema = build_rlds_schema(spec)
    
    # Convert to TFDS features format
    features = {
        "steps": {
            "observation": {},
            "action": _to_tfds_tensor(schema["episode"]["steps"]["action"]),
            "reward": _to_tfds_tensor(schema["episode"]["steps"]["reward"]),
            "discount": _to_tfds_tensor(schema["episode"]["steps"]["discount"]),
            "is_first": _to_tfds_tensor(schema["episode"]["steps"]["is_first"]),
            "is_last": _to_tfds_tensor(schema["episode"]["steps"]["is_last"]),
            "is_terminal": _to_tfds_tensor(schema["episode"]["steps"]["is_terminal"]),
        }
    }
    
    # Add observation features
    for key, feat in schema["episode"]["steps"]["observation"].items():
        features["steps"]["observation"][key] = _to_tfds_tensor(feat)
    
    return features


def _to_tfds_tensor(feat: dict[str, Any]) -> dict[str, Any]:
    """Convert feature dict to TFDS Tensor spec."""
    return {
        "type": "Tensor",
        "dtype": feat["dtype"],
        "shape": list(feat["shape"]) if feat["shape"] else [],
    }
