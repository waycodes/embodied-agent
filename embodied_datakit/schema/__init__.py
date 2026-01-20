"""Schema subpackage for EmbodiedDataKit."""

from embodied_datakit.schema.action import ActionMetadata, ActionType
from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.index import EpisodeIndexRecord
from embodied_datakit.schema.keys import (
    KEY_SEP,
    flatten_dict,
    get_camera_name,
    is_depth_key,
    is_image_key,
    make_depth_key,
    make_image_key,
    normalize_key,
    unflatten_dict,
)
from embodied_datakit.schema.spec import DatasetSpec, FeatureSpec
from embodied_datakit.schema.stats import DatasetStats, FeatureStats, StatsAccumulator
from embodied_datakit.schema.step import Step
from embodied_datakit.schema.tasks import TaskCatalog
from embodied_datakit.schema.versioning import (
    CURRENT_SCHEMA_VERSION,
    Version,
    can_read,
    check_compatibility,
    get_current_version,
)

__all__ = [
    "Step",
    "Episode",
    "DatasetSpec",
    "FeatureSpec",
    "TaskCatalog",
    "EpisodeIndexRecord",
    "FeatureStats",
    "DatasetStats",
    "StatsAccumulator",
    "ActionType",
    "ActionMetadata",
    "KEY_SEP",
    "flatten_dict",
    "unflatten_dict",
    "normalize_key",
    "is_image_key",
    "is_depth_key",
    "get_camera_name",
    "make_image_key",
    "make_depth_key",
    "CURRENT_SCHEMA_VERSION",
    "Version",
    "can_read",
    "check_compatibility",
    "get_current_version",
]
