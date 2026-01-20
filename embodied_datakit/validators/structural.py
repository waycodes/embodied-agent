"""Structural validators for RLDS invariants."""

from __future__ import annotations

import logging

import numpy as np

from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.spec import DatasetSpec
from embodied_datakit.validators.base import BaseValidator, Finding, Severity

logger = logging.getLogger(__name__)


class RLDSInvariantValidator(BaseValidator):
    """Validate RLDS structural invariants.

    Checks:
    - is_first is True only for first step
    - is_last is True only for last step
    - Action is None for last step
    - Terminal implies last
    - Episode has at least one step
    """

    def __init__(self) -> None:
        """Initialize RLDS invariant validator."""
        super().__init__("rlds_invariants")

    def validate_episode(
        self, episode: Episode, spec: DatasetSpec
    ) -> list[Finding]:
        """Validate RLDS invariants for episode."""
        findings: list[Finding] = []

        if not episode.steps:
            findings.append(Finding(
                validator=self.name,
                severity=Severity.ERROR,
                message="Episode has no steps",
                episode_id=episode.episode_id,
            ))
            return findings

        # Check first step
        first_step = episode.steps[0]
        if not first_step.is_first:
            findings.append(Finding(
                validator=self.name,
                severity=Severity.ERROR,
                message="First step must have is_first=True",
                episode_id=episode.episode_id,
                step_index=0,
            ))

        # Check last step
        last_step = episode.steps[-1]
        if not last_step.is_last:
            findings.append(Finding(
                validator=self.name,
                severity=Severity.ERROR,
                message="Last step must have is_last=True",
                episode_id=episode.episode_id,
                step_index=len(episode.steps) - 1,
            ))

        if last_step.action is not None:
            findings.append(Finding(
                validator=self.name,
                severity=Severity.WARN,
                message="Last step should have action=None (RLDS convention)",
                episode_id=episode.episode_id,
                step_index=len(episode.steps) - 1,
            ))

        # Check middle steps
        for i, step in enumerate(episode.steps[1:-1], start=1):
            if step.is_first:
                findings.append(Finding(
                    validator=self.name,
                    severity=Severity.ERROR,
                    message="Only first step should have is_first=True",
                    episode_id=episode.episode_id,
                    step_index=i,
                ))
            if step.is_last:
                findings.append(Finding(
                    validator=self.name,
                    severity=Severity.ERROR,
                    message="Only last step should have is_last=True",
                    episode_id=episode.episode_id,
                    step_index=i,
                ))

        # Check terminal implies last
        for i, step in enumerate(episode.steps):
            if step.is_terminal and not step.is_last:
                findings.append(Finding(
                    validator=self.name,
                    severity=Severity.WARN,
                    message="Terminal step is not marked as last",
                    episode_id=episode.episode_id,
                    step_index=i,
                ))

        return findings


class EpisodeLengthValidator(BaseValidator):
    """Validate episode length constraints."""

    def __init__(
        self,
        min_length: int = 1,
        max_length: int = 100000,
    ) -> None:
        """Initialize episode length validator.

        Args:
            min_length: Minimum allowed steps.
            max_length: Maximum allowed steps.
        """
        super().__init__("episode_length")
        self.min_length = min_length
        self.max_length = max_length

    def validate_episode(
        self, episode: Episode, spec: DatasetSpec
    ) -> list[Finding]:
        """Validate episode length."""
        findings: list[Finding] = []

        num_steps = len(episode.steps)

        if num_steps < self.min_length:
            findings.append(Finding(
                validator=self.name,
                severity=Severity.ERROR,
                message=f"Episode too short: {num_steps} < {self.min_length}",
                episode_id=episode.episode_id,
            ))

        if num_steps > self.max_length:
            findings.append(Finding(
                validator=self.name,
                severity=Severity.WARN,
                message=f"Episode very long: {num_steps} > {self.max_length}",
                episode_id=episode.episode_id,
            ))

        return findings


class TimestampValidator(BaseValidator):
    """Validate timestamp monotonicity and consistency."""

    def __init__(
        self,
        max_gap_factor: float = 2.0,
        control_rate_hz: float = 10.0,
    ) -> None:
        """Initialize timestamp validator.

        Args:
            max_gap_factor: Maximum allowed gap as factor of expected interval.
            control_rate_hz: Expected control frequency.
        """
        super().__init__("timestamps")
        self.max_gap_factor = max_gap_factor
        self.control_rate_hz = control_rate_hz

    def validate_episode(
        self, episode: Episode, spec: DatasetSpec
    ) -> list[Finding]:
        """Validate timestamps."""
        findings: list[Finding] = []

        if len(episode.steps) < 2:
            return findings

        # Get control rate from spec if available
        control_rate = spec.control_rate_hz or self.control_rate_hz
        expected_interval = 1.0 / control_rate
        max_gap = expected_interval * self.max_gap_factor

        prev_ts = episode.steps[0].timestamp
        for i, step in enumerate(episode.steps[1:], start=1):
            curr_ts = step.timestamp

            # Check monotonicity
            if curr_ts < prev_ts:
                findings.append(Finding(
                    validator=self.name,
                    severity=Severity.ERROR,
                    message=f"Non-monotonic timestamp: {curr_ts} < {prev_ts}",
                    episode_id=episode.episode_id,
                    step_index=i,
                ))

            # Check gap
            gap = curr_ts - prev_ts
            if gap > max_gap:
                findings.append(Finding(
                    validator=self.name,
                    severity=Severity.WARN,
                    message=f"Large timestamp gap: {gap:.3f}s (expected ~{expected_interval:.3f}s)",
                    episode_id=episode.episode_id,
                    step_index=i,
                ))

            prev_ts = curr_ts

        return findings


class ActionSanityValidator(BaseValidator):
    """Validate action values are reasonable."""

    def __init__(
        self,
        bounds: tuple[float, float] = (-10.0, 10.0),
        sigma_threshold: float = 5.0,
    ) -> None:
        """Initialize action sanity validator.

        Args:
            bounds: (min, max) absolute bounds for actions.
            sigma_threshold: Outlier threshold in standard deviations.
        """
        super().__init__("action_sanity")
        self.bounds = bounds
        self.sigma_threshold = sigma_threshold

    def validate_episode(
        self, episode: Episode, spec: DatasetSpec
    ) -> list[Finding]:
        """Validate action values."""
        findings: list[Finding] = []

        actions = []
        for step in episode.steps:
            if step.action is not None:
                actions.append(step.action)

        if not actions:
            return findings

        actions_arr = np.stack(actions)

        # Check bounds
        min_val, max_val = self.bounds
        for i, step in enumerate(episode.steps):
            if step.action is None:
                continue

            if np.any(step.action < min_val) or np.any(step.action > max_val):
                findings.append(Finding(
                    validator=self.name,
                    severity=Severity.WARN,
                    message=f"Action out of bounds [{min_val}, {max_val}]",
                    episode_id=episode.episode_id,
                    step_index=i,
                ))
                break  # Only report once per episode

        # Check for NaN/Inf
        if np.any(~np.isfinite(actions_arr)):
            findings.append(Finding(
                validator=self.name,
                severity=Severity.ERROR,
                message="Action contains NaN or Inf",
                episode_id=episode.episode_id,
            ))

        # Check for outliers
        mean = np.mean(actions_arr, axis=0)
        std = np.std(actions_arr, axis=0)
        std = np.where(std < 1e-8, 1.0, std)  # Avoid division by zero

        for i, step in enumerate(episode.steps):
            if step.action is None:
                continue

            z_score = np.abs((step.action - mean) / std)
            if np.any(z_score > self.sigma_threshold):
                findings.append(Finding(
                    validator=self.name,
                    severity=Severity.WARN,
                    message=f"Action outlier detected (z > {self.sigma_threshold})",
                    episode_id=episode.episode_id,
                    step_index=i,
                ))
                break

        return findings
