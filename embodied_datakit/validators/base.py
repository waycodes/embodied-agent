"""Base validator interface for episode validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.spec import DatasetSpec


class Severity(str, Enum):
    """Validation finding severity levels."""

    ERROR = "ERROR"    # Cannot compile; fatal violation
    WARN = "WARN"      # Compile but mark invalid
    INFO = "INFO"      # Statistics only; no action


@dataclass
class Finding:
    """A single validation finding.

    Attributes:
        severity: Severity level.
        code: Validation rule code (e.g., "E001", "W101").
        message: Human-readable message.
        location: Where the issue occurred (e.g., "step 5", "observation.images.front").
        field: Affected field name.
        value: Problematic value (for debugging).
        episode_id: Episode where finding occurred.
        step_index: Step index if applicable.
    """

    severity: Severity
    code: str
    message: str
    location: str | None = None
    field: str | None = None
    value: Any = None
    episode_id: str | None = None
    step_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "field": self.field,
            "value": str(self.value) if self.value is not None else None,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
        }


@runtime_checkable
class Validator(Protocol):
    """Protocol for episode validators."""

    @property
    def name(self) -> str:
        """Validator name."""
        ...

    def validate_episode(self, episode: Episode, spec: DatasetSpec) -> list[Finding]:
        """Validate an episode.

        Args:
            episode: Episode to validate.
            spec: Dataset specification.

        Returns:
            List of findings (empty if valid).
        """
        ...


class BaseValidator(ABC):
    """Abstract base class for validators."""

    @property
    def name(self) -> str:
        """Validator name (defaults to class name)."""
        return self.__class__.__name__

    @abstractmethod
    def validate_episode(self, episode: Episode, spec: DatasetSpec) -> list[Finding]:
        """Validate an episode."""
        pass

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}()"


@dataclass
class ValidationReport:
    """Aggregated validation report.

    Attributes:
        total_episodes: Total episodes validated.
        valid_episodes: Episodes with no errors.
        invalid_episodes: Episodes with warnings (marked invalid).
        error_episodes: Episodes with errors (rejected).
        findings: All findings.
        by_severity: Findings grouped by severity.
        by_code: Findings grouped by code.
    """

    total_episodes: int = 0
    valid_episodes: int = 0
    invalid_episodes: int = 0
    error_episodes: int = 0
    findings: list[Finding] = field(default_factory=list)

    def add_finding(self, finding: Finding) -> None:
        """Add a finding to the report."""
        self.findings.append(finding)

    def add_episode_result(self, findings: list[Finding]) -> None:
        """Add results for an episode."""
        self.total_episodes += 1

        has_error = any(f.severity == Severity.ERROR for f in findings)
        has_warn = any(f.severity == Severity.WARN for f in findings)

        if has_error:
            self.error_episodes += 1
        elif has_warn:
            self.invalid_episodes += 1
        else:
            self.valid_episodes += 1

        self.findings.extend(findings)

    @property
    def error_count(self) -> int:
        """Count of ERROR-level findings."""
        return sum(1 for f in self.findings if f.severity == Severity.ERROR)

    @property
    def warn_count(self) -> int:
        """Count of WARN-level findings."""
        return sum(1 for f in self.findings if f.severity == Severity.WARN)

    @property
    def info_count(self) -> int:
        """Count of INFO-level findings."""
        return sum(1 for f in self.findings if f.severity == Severity.INFO)

    def by_severity(self) -> dict[Severity, list[Finding]]:
        """Group findings by severity."""
        result: dict[Severity, list[Finding]] = {s: [] for s in Severity}
        for finding in self.findings:
            result[finding.severity].append(finding)
        return result

    def by_code(self) -> dict[str, list[Finding]]:
        """Group findings by code."""
        result: dict[str, list[Finding]] = {}
        for finding in self.findings:
            if finding.code not in result:
                result[finding.code] = []
            result[finding.code].append(finding)
        return result

    def summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        by_code = self.by_code()
        return {
            "total_episodes": self.total_episodes,
            "valid_episodes": self.valid_episodes,
            "invalid_episodes": self.invalid_episodes,
            "error_episodes": self.error_episodes,
            "error_count": self.error_count,
            "warn_count": self.warn_count,
            "info_count": self.info_count,
            "by_severity": {
                "ERROR": {code: len(fs) for code, fs in by_code.items()
                         if fs and fs[0].severity == Severity.ERROR},
                "WARN": {code: len(fs) for code, fs in by_code.items()
                        if fs and fs[0].severity == Severity.WARN},
                "INFO": {code: len(fs) for code, fs in by_code.items()
                        if fs and fs[0].severity == Severity.INFO},
            },
        }

    def has_errors(self) -> bool:
        """Check if any errors exist."""
        return self.error_count > 0

    def has_warnings(self) -> bool:
        """Check if any warnings exist."""
        return self.warn_count > 0
