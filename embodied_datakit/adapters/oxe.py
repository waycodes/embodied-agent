"""Open X-Embodiment specific utilities and dataset locator."""

from __future__ import annotations

from dataclasses import dataclass

# Known Open X-Embodiment datasets
# Source: https://github.com/google-deepmind/open_x_embodiment
OXE_DATASETS: dict[str, dict[str, str | list[str]]] = {
    "berkeley_autolab_ur5": {
        "display_name": "Berkeley Autolab UR5",
        "robot": "UR5",
        "cameras": ["cam_high", "cam_low"],
    },
    "bridge": {
        "display_name": "Bridge Data",
        "robot": "WidowX",
        "cameras": ["image_0", "image_1", "image_2", "image_3", "image_4", "image_5"],
    },
    "taco_play": {
        "display_name": "TACO Play",
        "robot": "Franka",
        "cameras": ["rgb_static", "rgb_gripper"],
    },
    "jaco_play": {
        "display_name": "Jaco Play",
        "robot": "Jaco",
        "cameras": ["image"],
    },
    "berkeley_cable_routing": {
        "display_name": "Berkeley Cable Routing",
        "robot": "Franka",
        "cameras": ["image", "wrist_image"],
    },
    "roboturk": {
        "display_name": "RoboTurk",
        "robot": "Sawyer",
        "cameras": ["front_rgb"],
    },
    "nyu_door_opening_surprising_effectiveness": {
        "display_name": "NYU Door Opening",
        "robot": "Hello Robot Stretch",
        "cameras": ["image"],
    },
    "viola": {
        "display_name": "VIOLA",
        "robot": "Franka",
        "cameras": ["agentview_rgb", "eye_in_hand_rgb"],
    },
    "berkeley_fanuc_manipulation": {
        "display_name": "Berkeley Fanuc",
        "robot": "Fanuc LR Mate",
        "cameras": ["image"],
    },
    "language_table": {
        "display_name": "Language Table",
        "robot": "xArm + Custom Gripper",
        "cameras": ["rgb"],
    },
    "droid": {
        "display_name": "DROID",
        "robot": "Franka",
        "cameras": ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"],
    },
    "aloha_dagger_dataset": {
        "display_name": "ALOHA DAgger",
        "robot": "ViperX (Bimanual)",
        "cameras": ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"],
    },
}


@dataclass
class OXEDatasetInfo:
    """Information about an Open X-Embodiment dataset."""

    name: str
    display_name: str
    robot: str
    cameras: list[str]


def get_oxe_dataset_info(name: str) -> OXEDatasetInfo | None:
    """Get information about an OXE dataset.

    Args:
        name: Dataset name.

    Returns:
        OXEDatasetInfo or None if not found.
    """
    if name not in OXE_DATASETS:
        return None

    info = OXE_DATASETS[name]
    cameras = info.get("cameras", [])
    return OXEDatasetInfo(
        name=name,
        display_name=str(info.get("display_name", name)),
        robot=str(info.get("robot", "unknown")),
        cameras=cameras if isinstance(cameras, list) else [],
    )


def list_oxe_datasets() -> list[str]:
    """List all known OXE dataset names."""
    return list(OXE_DATASETS.keys())


def get_recommended_camera(name: str) -> str | None:
    """Get the recommended canonical camera for an OXE dataset.

    Prefers cameras named: front, cam_high, image, image_0, exterior_image_1_left

    Args:
        name: Dataset name.

    Returns:
        Recommended camera name or None.
    """
    info = get_oxe_dataset_info(name)
    if info is None:
        return None

    cameras = info.cameras
    if not cameras:
        return None

    # Priority order for canonical camera selection
    priority = ["front", "cam_high", "image", "rgb", "agentview_rgb", "exterior_image_1_left"]

    for p in priority:
        if p in cameras:
            return p

    # Fall back to first camera
    return cameras[0]
