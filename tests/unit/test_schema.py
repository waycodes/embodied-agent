"""Unit tests for schema module."""

import numpy as np
import pytest

from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.spec import DatasetSpec, FeatureSpec
from embodied_datakit.schema.stats import DatasetStats, FeatureStats, StatsAccumulator
from embodied_datakit.schema.step import Step
from embodied_datakit.schema.tasks import TaskCatalog


class TestStep:
    """Tests for Step dataclass."""

    def test_create_step(self) -> None:
        """Test creating a basic step."""
        step = Step(
            is_first=True,
            is_last=False,
            observation={"observation.state": np.array([1.0, 2.0])},
            action=np.array([0.1, 0.2]),
        )
        assert step.is_first is True
        assert step.is_last is False
        assert step.has_valid_action is True

    def test_step_last_no_valid_action(self) -> None:
        """Test that last step has no valid action."""
        step = Step(
            is_first=False,
            is_last=True,
            observation={"observation.state": np.array([1.0])},
            action=None,
        )
        assert step.has_valid_action is False

    def test_step_to_dict_from_dict(self) -> None:
        """Test serialization round-trip."""
        step = Step(
            is_first=True,
            is_last=False,
            observation={"observation.state": np.array([1.0, 2.0])},
            action=np.array([0.1, 0.2]),
            reward=0.5,
        )
        data = step.to_dict()
        restored = Step.from_dict(data)
        assert restored.is_first == step.is_first
        assert restored.reward == step.reward


class TestEpisode:
    """Tests for Episode dataclass."""

    def test_episode_invariants(self, sample_episode: Episode) -> None:
        """Test RLDS invariants are satisfied."""
        assert sample_episode.steps[0].is_first is True
        assert sample_episode.steps[-1].is_last is True
        issues = sample_episode.validate_structure()
        assert len(issues) == 0

    def test_episode_num_steps(self, sample_episode: Episode) -> None:
        """Test step counting."""
        assert sample_episode.num_steps == 10

    def test_episode_duration(self, sample_episode: Episode) -> None:
        """Test duration calculation."""
        assert sample_episode.duration == pytest.approx(0.9, abs=0.01)

    def test_episode_get_camera_names(self, sample_episode: Episode) -> None:
        """Test camera name extraction."""
        cameras = sample_episode.get_camera_names()
        assert "front" in cameras


class TestDatasetSpec:
    """Tests for DatasetSpec."""

    def test_spec_to_dict_from_dict(self, sample_spec: DatasetSpec) -> None:
        """Test serialization round-trip."""
        data = sample_spec.to_dict()
        restored = DatasetSpec.from_dict(data)
        assert restored.dataset_id == sample_spec.dataset_id
        assert restored.control_rate_hz == sample_spec.control_rate_hz

    def test_spec_num_cameras(self, sample_spec: DatasetSpec) -> None:
        """Test camera count."""
        assert sample_spec.num_cameras == 1

    def test_spec_has_video(self, sample_spec: DatasetSpec) -> None:
        """Test video detection."""
        assert sample_spec.has_video is True


class TestTaskCatalog:
    """Tests for TaskCatalog."""

    def test_add_and_get(self) -> None:
        """Test adding and retrieving tasks."""
        catalog = TaskCatalog()
        id1 = catalog.add("Pick up red block")
        id2 = catalog.add("Pick up blue block")
        assert id1 == 0
        assert id2 == 1
        assert catalog.get_task(0) == "Pick up red block"

    def test_idempotent_add(self) -> None:
        """Test that adding same task returns same ID."""
        catalog = TaskCatalog()
        id1 = catalog.add("Pick up red block")
        id2 = catalog.add("Pick up red block")
        assert id1 == id2


class TestStats:
    """Tests for feature statistics."""

    def test_accumulator(self) -> None:
        """Test stats accumulation."""
        acc = StatsAccumulator()
        acc.add("action", np.array([1.0, 2.0, 3.0]))
        acc.add("action", np.array([2.0, 3.0, 4.0]))
        stats = acc.compute()
        assert "action" in stats
        assert stats["action"].count == 2
        assert stats["action"].mean == pytest.approx([1.5, 2.5, 3.5], abs=0.01)

    def test_normalize_denormalize(self) -> None:
        """Test normalization round-trip."""
        stats = FeatureStats(
            mean=[0.0, 1.0],
            std=[1.0, 2.0],
            min=[-1.0, -1.0],
            max=[1.0, 3.0],
            count=100,
        )
        value = np.array([0.0, 1.0])
        normalized = stats.normalize(value)
        denormalized = stats.denormalize(normalized)
        assert np.allclose(value, denormalized)
