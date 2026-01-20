"""Timestamp canonicalization and resampling transforms."""

from __future__ import annotations

import numpy as np

from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.spec import DatasetSpec
from embodied_datakit.schema.step import Step
from embodied_datakit.transforms.base import BaseTransform


class TimestampCanonicalizeTransform(BaseTransform):
    """Ensure monotonic timestamps; synthesize from control rate if missing."""

    def __init__(self, control_rate_hz: float = 10.0) -> None:
        """Initialize timestamp canonicalization.

        Args:
            control_rate_hz: Control rate for synthesizing timestamps.
        """
        super().__init__("timestamp_canonicalize")
        self.control_rate_hz = control_rate_hz
        self.dt = 1.0 / control_rate_hz

    def transform_episode(self, episode: Episode, spec: DatasetSpec) -> Episode:
        """Ensure all steps have monotonic timestamps."""
        rate = spec.control_rate_hz if spec.control_rate_hz else self.control_rate_hz
        dt = 1.0 / rate

        new_steps = []
        for i, step in enumerate(episode.steps):
            ts = step.timestamp if step.timestamp is not None else i * dt
            new_steps.append(Step(
                is_first=step.is_first,
                is_last=step.is_last,
                is_terminal=step.is_terminal,
                observation=step.observation,
                action=step.action,
                reward=step.reward,
                discount=step.discount,
                timestamp=ts,
                step_metadata=step.step_metadata,
            ))

        # Enforce monotonicity
        for i in range(1, len(new_steps)):
            if new_steps[i].timestamp <= new_steps[i - 1].timestamp:
                new_steps[i] = Step(
                    is_first=new_steps[i].is_first,
                    is_last=new_steps[i].is_last,
                    is_terminal=new_steps[i].is_terminal,
                    observation=new_steps[i].observation,
                    action=new_steps[i].action,
                    reward=new_steps[i].reward,
                    discount=new_steps[i].discount,
                    timestamp=new_steps[i - 1].timestamp + dt,
                    step_metadata=new_steps[i].step_metadata,
                )

        return Episode(
            episode_id=episode.episode_id,
            dataset_id=episode.dataset_id,
            steps=new_steps,
            task_id=episode.task_id,
            task_text=episode.task_text,
            invalid=episode.invalid,
            episode_metadata=episode.episode_metadata,
        )


class ResampleTransform(BaseTransform):
    """Resample episode to target control rate."""

    def __init__(self, target_rate_hz: float = 10.0) -> None:
        """Initialize resampling transform.

        Args:
            target_rate_hz: Target control rate in Hz.
        """
        super().__init__("resample")
        self.target_rate_hz = target_rate_hz

    def transform_episode(self, episode: Episode, spec: DatasetSpec) -> Episode:
        """Resample episode to target rate."""
        if not episode.steps:
            return episode

        # Get timestamps
        timestamps = np.array([s.timestamp or i * 0.1 for i, s in enumerate(episode.steps)])
        duration = timestamps[-1] - timestamps[0]
        if duration <= 0:
            return episode

        # Generate target timestamps
        dt = 1.0 / self.target_rate_hz
        target_ts = np.arange(timestamps[0], timestamps[-1], dt)
        if len(target_ts) == 0:
            return episode

        # Nearest neighbor resampling
        new_steps = []
        for t in target_ts:
            idx = int(np.argmin(np.abs(timestamps - t)))
            src = episode.steps[idx]
            new_steps.append(Step(
                is_first=len(new_steps) == 0,
                is_last=False,
                is_terminal=False,
                observation=src.observation,
                action=src.action,
                reward=src.reward,
                discount=src.discount,
                timestamp=float(t),
                step_metadata=src.step_metadata,
            ))

        # Mark last step
        if new_steps:
            last = new_steps[-1]
            new_steps[-1] = Step(
                is_first=last.is_first,
                is_last=True,
                is_terminal=episode.steps[-1].is_terminal,
                observation=last.observation,
                action=last.action,
                reward=last.reward,
                discount=last.discount,
                timestamp=last.timestamp,
                step_metadata=last.step_metadata,
            )

        return Episode(
            episode_id=episode.episode_id,
            dataset_id=episode.dataset_id,
            steps=new_steps,
            task_id=episode.task_id,
            task_text=episode.task_text,
            invalid=episode.invalid,
            episode_metadata=episode.episode_metadata,
        )
