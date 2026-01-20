"""Synthetic fixture generator for testing."""

from __future__ import annotations

import numpy as np

from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.spec import DatasetSpec, FeatureSpec
from embodied_datakit.schema.step import Step


def generate_synthetic_episode(
    episode_idx: int = 0,
    num_steps: int = 10,
    image_size: tuple[int, int, int] = (64, 64, 3),
    state_dim: int = 7,
    action_dim: int = 7,
    seed: int | None = None,
    task_text: str = "pick up the red block",
) -> Episode:
    """Generate a synthetic episode for testing.

    Args:
        episode_idx: Episode index for ID generation.
        num_steps: Number of steps in episode.
        image_size: Image observation shape (H, W, C).
        state_dim: State observation dimension.
        action_dim: Action dimension.
        seed: Random seed for reproducibility.
        task_text: Task description.

    Returns:
        Synthetic Episode with deterministic data.
    """
    if seed is not None:
        np.random.seed(seed + episode_idx)

    steps = []
    for step_idx in range(num_steps):
        is_first = step_idx == 0
        is_last = step_idx == num_steps - 1

        # Generate observations
        observation = {
            "observation.images.front": np.random.randint(
                0, 256, size=image_size, dtype=np.uint8
            ),
            "observation.state": np.random.randn(state_dim).astype(np.float32),
            "observation.language": task_text,
        }

        # Generate action (None for last step per RLDS convention)
        action = None
        if not is_last:
            action = np.random.randn(action_dim).astype(np.float32) * 0.1

        step = Step(
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_last,
            observation=observation,
            action=action,
            reward=1.0 if is_last else 0.0,
            discount=0.0 if is_last else 0.99,
            timestamp=step_idx * 0.1,
        )
        steps.append(step)

    return Episode(
        episode_id=f"synthetic_{episode_idx:06d}",
        dataset_id="synthetic_test",
        steps=steps,
        task_text=task_text,
    )


def generate_synthetic_spec(
    image_size: tuple[int, int, int] = (64, 64, 3),
    state_dim: int = 7,
    action_dim: int = 7,
) -> DatasetSpec:
    """Generate a DatasetSpec matching synthetic episodes.

    Args:
        image_size: Image observation shape.
        state_dim: State observation dimension.
        action_dim: Action dimension.

    Returns:
        DatasetSpec for synthetic data.
    """
    return DatasetSpec(
        dataset_id="synthetic_test",
        dataset_name="Synthetic Test Dataset",
        observation_schema={
            "observation.images.front": FeatureSpec(
                dtype="uint8",
                shape=image_size,
                description="Front camera RGB",
                is_video=True,
            ),
            "observation.state": FeatureSpec(
                dtype="float32",
                shape=(state_dim,),
                description="Robot state",
            ),
            "observation.language": FeatureSpec(
                dtype="string",
                shape=(),
                description="Task instruction",
            ),
        },
        action_schema=FeatureSpec(
            dtype="float32",
            shape=(action_dim,),
            description="Action vector",
        ),
        control_rate_hz=10.0,
        action_space_type="ee_delta_7",
        camera_names=["front"],
        canonical_camera="front",
    )


def generate_synthetic_dataset(
    num_episodes: int = 5,
    steps_per_episode: int = 10,
    seed: int = 42,
) -> tuple[list[Episode], DatasetSpec]:
    """Generate a complete synthetic dataset.

    Args:
        num_episodes: Number of episodes to generate.
        steps_per_episode: Steps per episode.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (episodes list, DatasetSpec).
    """
    spec = generate_synthetic_spec()
    episodes = [
        generate_synthetic_episode(
            episode_idx=i,
            num_steps=steps_per_episode,
            seed=seed,
        )
        for i in range(num_episodes)
    ]
    return episodes, spec
