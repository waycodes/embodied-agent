"""Unit tests for transforms."""

import numpy as np
import pytest

from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.spec import DatasetSpec, FeatureSpec
from embodied_datakit.schema.step import Step
from embodied_datakit.transforms.action import (
    MapActionSpaceTransform,
    NormalizeActionsTransform,
    PadActionTransform,
)
from embodied_datakit.transforms.camera import ResizeImagesTransform, SelectCameraTransform


@pytest.fixture
def episode_with_cameras() -> Episode:
    """Create episode with multiple cameras."""
    steps = []
    for i in range(5):
        steps.append(Step(
            is_first=i == 0,
            is_last=i == 4,
            observation={
                "observation.images.front": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                "observation.images.wrist": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                "observation.state": np.random.randn(7).astype(np.float32),
            },
            action=np.random.randn(7).astype(np.float32) if i < 4 else None,
        ))
    return Episode(
        episode_id="test_001",
        dataset_id="test",
        steps=steps,
    )


@pytest.fixture
def spec_with_cameras() -> DatasetSpec:
    """Create spec with camera info."""
    return DatasetSpec(
        dataset_id="test",
        dataset_name="Test",
        observation_schema={
            "observation.images.front": FeatureSpec(dtype="uint8", shape=(64, 64, 3), is_video=True),
            "observation.images.wrist": FeatureSpec(dtype="uint8", shape=(64, 64, 3), is_video=True),
        },
        camera_names=["front", "wrist"],
    )


class TestSelectCameraTransform:
    """Tests for camera selection transform."""

    def test_select_specific_camera(
        self, episode_with_cameras: Episode, spec_with_cameras: DatasetSpec
    ) -> None:
        """Test selecting a specific camera."""
        transform = SelectCameraTransform(camera_name="front")
        result = transform.transform_episode(episode_with_cameras, spec_with_cameras)

        # Check canonical camera was added
        assert "observation.images.canonical" in result.steps[0].observation

    def test_fallback_order(
        self, episode_with_cameras: Episode, spec_with_cameras: DatasetSpec
    ) -> None:
        """Test fallback order for camera selection."""
        transform = SelectCameraTransform(fallback_order=["wrist", "front"])
        result = transform.transform_episode(episode_with_cameras, spec_with_cameras)

        # Should select wrist (first in fallback order that exists)
        assert transform._selected_camera == "wrist"


class TestResizeImagesTransform:
    """Tests for image resize transform."""

    def test_resize_images(
        self, episode_with_cameras: Episode, spec_with_cameras: DatasetSpec
    ) -> None:
        """Test resizing images."""
        transform = ResizeImagesTransform(target_size=(32, 32))
        result = transform.transform_episode(episode_with_cameras, spec_with_cameras)

        # Check images were resized
        front = result.steps[0].observation["observation.images.front"]
        assert front.shape == (32, 32, 3)


class TestNormalizeActionsTransform:
    """Tests for action normalization transform."""

    def test_normalize_with_bounds(
        self, episode_with_cameras: Episode, spec_with_cameras: DatasetSpec
    ) -> None:
        """Test normalizing with explicit bounds."""
        transform = NormalizeActionsTransform(action_bounds=(-1.0, 1.0))
        result = transform.transform_episode(episode_with_cameras, spec_with_cameras)

        # Check actions are within [-1, 1]
        for step in result.steps:
            if step.action is not None:
                assert np.all(step.action >= -1.0)
                assert np.all(step.action <= 1.0)

    def test_normalize_infers_bounds(
        self, episode_with_cameras: Episode, spec_with_cameras: DatasetSpec
    ) -> None:
        """Test normalization with inferred bounds."""
        transform = NormalizeActionsTransform()
        result = transform.transform_episode(episode_with_cameras, spec_with_cameras)

        # Should complete without error
        assert len(result.steps) == len(episode_with_cameras.steps)


class TestPadActionTransform:
    """Tests for action padding transform."""

    def test_pad_action(self) -> None:
        """Test padding action to target dimension."""
        steps = [
            Step(
                is_first=True,
                is_last=True,
                observation={},
                action=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            )
        ]
        episode = Episode(episode_id="test", dataset_id="test", steps=steps)
        spec = DatasetSpec(dataset_id="test", dataset_name="Test")

        transform = PadActionTransform(target_dim=7, pad_value=0.0)
        result = transform.transform_episode(episode, spec)

        assert result.steps[0].action is not None
        assert len(result.steps[0].action) == 7
        assert np.allclose(result.steps[0].action[:3], [1.0, 2.0, 3.0])
        assert np.allclose(result.steps[0].action[3:], [0.0, 0.0, 0.0, 0.0])

    def test_truncate_action(self) -> None:
        """Test truncating action to target dimension."""
        steps = [
            Step(
                is_first=True,
                is_last=True,
                observation={},
                action=np.arange(10, dtype=np.float32),
            )
        ]
        episode = Episode(episode_id="test", dataset_id="test", steps=steps)
        spec = DatasetSpec(dataset_id="test", dataset_name="Test")

        transform = PadActionTransform(target_dim=7)
        result = transform.transform_episode(episode, spec)

        assert result.steps[0].action is not None
        assert len(result.steps[0].action) == 7


class TestMapActionSpaceTransform:
    """Tests for action space mapping."""

    def test_map_joint_to_ee(self) -> None:
        """Test mapping joint space to end-effector space."""
        steps = [
            Step(
                is_first=True,
                is_last=True,
                observation={},
                action=np.arange(8, dtype=np.float32),  # 7 joints + gripper
            )
        ]
        episode = Episode(episode_id="test", dataset_id="test", steps=steps)
        spec = DatasetSpec(dataset_id="test", dataset_name="Test")

        transform = MapActionSpaceTransform(
            source_space="joint_7_gripper",
            target_space="ee_delta_7",
        )
        result = transform.transform_episode(episode, spec)

        assert result.steps[0].action is not None
        assert len(result.steps[0].action) == 7
