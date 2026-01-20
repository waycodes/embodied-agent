"""RLDS/TFDS writer subpackage."""

from embodied_datakit.writers.rlds_tfds.schema import (
    build_rlds_schema,
    build_tfds_features_dict,
)

# TFRecordShardWriter requires TensorFlow
try:
    from embodied_datakit.writers.rlds_tfds.writer import TFRecordShardWriter
    __all__ = ["build_rlds_schema", "build_tfds_features_dict", "TFRecordShardWriter"]
except ImportError:
    __all__ = ["build_rlds_schema", "build_tfds_features_dict"]
