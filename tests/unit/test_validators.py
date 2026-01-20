"""Unit tests for validators."""

import numpy as np
import pytest

from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.spec import DatasetSpec
from embodied_datakit.schema.step import Step
from embodied_datakit.validators.base import Severity
from embodied_datakit.validators.structural import (
    ActionSanityValidator,
    EpisodeLengthValidator,
    RLDSInvariantValidator,
    TimestampValidator,
)


@pytest.fixture
def valid_episode() -> Episode:
    """Create a valid episode that passes all validators."""
    steps = []
    for i in range(10):
        steps.append(Step(
            is_first=i == 0,
            is_last=i == 9,
            observation={"observation.state": np.zeros(7, dtype=np.float32)},
            action=np.zeros(7, dtype=np.float32) if i < 9 else None,
            timestamp=i * 0.1,
        ))
    return Episode(
        episode_id="valid_001",
        dataset_id="test",
        steps=steps,
    )


@pytest.fixture
def spec() -> DatasetSpec:
    """Create a basic spec."""
    return DatasetSpec(
        dataset_id="test",
        dataset_name="Test",
        control_rate_hz=10.0,
    )


class TestRLDSInvariantValidator:
    """Tests for RLDS invariant validation."""

    def test_valid_episode(self, valid_episode: Episode, spec: DatasetSpec) -> None:
        """Test validation passes for valid episode."""
        validator = RLDSInvariantValidator()
        findings = validator.validate_episode(valid_episode, spec)

        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_missing_is_first(self, spec: DatasetSpec) -> None:
        """Test detection of missing is_first."""
        steps = [
            Step(is_first=False, is_last=False, observation={}, action=np.zeros(7)),
            Step(is_first=False, is_last=True, observation={}, action=None),
        ]
        episode = Episode(episode_id="bad", dataset_id="test", steps=steps, invalid=True)

        validator = RLDSInvariantValidator()
        findings = validator.validate_episode(episode, spec)

        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) > 0
        assert any("is_first" in f.message for f in errors)

    def test_missing_is_last(self, spec: DatasetSpec) -> None:
        """Test detection of missing is_last."""
        steps = [
            Step(is_first=True, is_last=False, observation={}, action=np.zeros(7)),
            Step(is_first=False, is_last=False, observation={}, action=np.zeros(7)),
        ]
        episode = Episode(episode_id="bad", dataset_id="test", steps=steps, invalid=True)

        validator = RLDSInvariantValidator()
        findings = validator.validate_episode(episode, spec)

        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) > 0
        assert any("is_last" in f.message for f in errors)

    def test_empty_episode(self, spec: DatasetSpec) -> None:
        """Test detection of empty episode."""
        episode = Episode(episode_id="empty", dataset_id="test", steps=[])

        validator = RLDSInvariantValidator()
        findings = validator.validate_episode(episode, spec)

        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) > 0


class TestEpisodeLengthValidator:
    """Tests for episode length validation."""

    def test_valid_length(self, valid_episode: Episode, spec: DatasetSpec) -> None:
        """Test validation passes for valid length."""
        validator = EpisodeLengthValidator(min_length=1, max_length=100)
        findings = validator.validate_episode(valid_episode, spec)

        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_too_short(self, spec: DatasetSpec) -> None:
        """Test detection of too-short episode."""
        steps = [Step(is_first=True, is_last=True, observation={}, action=None)]
        episode = Episode(episode_id="short", dataset_id="test", steps=steps)

        validator = EpisodeLengthValidator(min_length=5)
        findings = validator.validate_episode(episode, spec)

        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) > 0


class TestTimestampValidator:
    """Tests for timestamp validation."""

    def test_valid_timestamps(self, valid_episode: Episode, spec: DatasetSpec) -> None:
        """Test validation passes for valid timestamps."""
        validator = TimestampValidator()
        findings = validator.validate_episode(valid_episode, spec)

        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_non_monotonic(self, spec: DatasetSpec) -> None:
        """Test detection of non-monotonic timestamps."""
        steps = [
            Step(is_first=True, is_last=False, observation={}, action=np.zeros(7), timestamp=0.0),
            Step(is_first=False, is_last=False, observation={}, action=np.zeros(7), timestamp=0.2),
            Step(is_first=False, is_last=True, observation={}, action=None, timestamp=0.1),  # Out of order
        ]
        episode = Episode(episode_id="bad_ts", dataset_id="test", steps=steps)

        validator = TimestampValidator()
        findings = validator.validate_episode(episode, spec)

        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) > 0


class TestActionSanityValidator:
    """Tests for action sanity validation."""

    def test_valid_actions(self, valid_episode: Episode, spec: DatasetSpec) -> None:
        """Test validation passes for valid actions."""
        validator = ActionSanityValidator()
        findings = validator.validate_episode(valid_episode, spec)

        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_nan_action(self, spec: DatasetSpec) -> None:
        """Test detection of NaN in actions."""
        steps = [
            Step(is_first=True, is_last=False, observation={}, action=np.array([np.nan, 0, 0])),
            Step(is_first=False, is_last=True, observation={}, action=None),
        ]
        episode = Episode(episode_id="nan_action", dataset_id="test", steps=steps)

        validator = ActionSanityValidator()
        findings = validator.validate_episode(episode, spec)

        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) > 0
        assert any("NaN" in f.message for f in errors)
