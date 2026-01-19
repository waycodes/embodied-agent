"""Test fixtures and conftest."""

import numpy as np
import pytest

from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.spec import DatasetSpec, FeatureSpec
from embodied_datakit.schema.step import Step


@pytest.fixture
def sample_step() -> Step:
    """Create a sample step for testing."""
    return Step(
        is_first=True,
        is_last=False,
        observation={
            "observation.images.front": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            "observation.state": np.random.randn(7).astype(np.float32),
        },
        action=np.random.randn(7).astype(np.float32),
        reward=0.0,
        discount=1.0,
        timestamp=0.0,
    )


@pytest.fixture
def sample_episode() -> Episode:
    """Create a sample episode for testing."""
    steps = []

    # First step
    steps.append(Step(
        is_first=True,
        is_last=False,
        observation={
            "observation.images.front": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            "observation.state": np.random.randn(7).astype(np.float32),
        },
        action=np.random.randn(7).astype(np.float32),
        reward=0.0,
        discount=1.0,
        timestamp=0.0,
    ))

    # Middle steps
    for i in range(1, 9):
        steps.append(Step(
            is_first=False,
            is_last=False,
            observation={
                "observation.images.front": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                "observation.state": np.random.randn(7).astype(np.float32),
            },
            action=np.random.randn(7).astype(np.float32),
            reward=0.0,
            discount=1.0,
            timestamp=float(i) * 0.1,
        ))

    # Last step
    steps.append(Step(
        is_first=False,
        is_last=True,
        is_terminal=True,
        observation={
            "observation.images.front": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            "observation.state": np.random.randn(7).astype(np.float32),
        },
        action=None,
        reward=1.0,
        discount=0.0,
        timestamp=0.9,
    ))

    return Episode(
        episode_id="test_episode_001",
        dataset_id="test_dataset",
        steps=steps,
        task_id=0,
        task_text="Pick up the red block",
    )


@pytest.fixture
def sample_spec() -> DatasetSpec:
    """Create a sample dataset spec for testing."""
    return DatasetSpec(
        dataset_id="test_dataset",
        dataset_name="Test Dataset",
        observation_schema={
            "observation.images.front": FeatureSpec(
                dtype="uint8",
                shape=(256, 256, 3),
                description="Front camera RGB",
                is_video=True,
            ),
            "observation.state": FeatureSpec(
                dtype="float32",
                shape=(7,),
                description="Proprioceptive state",
            ),
        },
        action_schema=FeatureSpec(
            dtype="float32",
            shape=(7,),
            description="7D end-effector action",
        ),
        control_rate_hz=10.0,
        action_space_type="ee_delta_7",
        camera_names=["front"],
        canonical_camera="front",
    )
