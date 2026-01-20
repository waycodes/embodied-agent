"""Validation report generation and execution modes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.spec import DatasetSpec
from embodied_datakit.validators.base import (
    BaseValidator,
    Finding,
    Severity,
    ValidationReport,
)


class ExecutionMode(str, Enum):
    """Validation execution modes."""
    
    FAIL_FAST = "fail_fast"      # Stop on first error
    QUARANTINE = "quarantine"    # Continue, partition invalid episodes
    COLLECT = "collect"          # Collect all findings, don't stop


@dataclass
class ValidationResult:
    """Result of validating a single episode."""
    
    episode_id: str
    findings: list[Finding]
    is_valid: bool
    is_quarantined: bool


class ValidationRunner:
    """Run validators with configurable execution mode."""
    
    def __init__(
        self,
        validators: list[BaseValidator],
        mode: ExecutionMode = ExecutionMode.COLLECT,
        fail_on_warn: bool = False,
    ) -> None:
        """Initialize validation runner.
        
        Args:
            validators: List of validators to run.
            mode: Execution mode.
            fail_on_warn: Treat warnings as errors in fail_fast mode.
        """
        self.validators = validators
        self.mode = mode
        self.fail_on_warn = fail_on_warn
        self.report = ValidationReport()
    
    def validate_episode(
        self, episode: Episode, spec: DatasetSpec
    ) -> ValidationResult:
        """Validate a single episode.
        
        Args:
            episode: Episode to validate.
            spec: Dataset specification.
        
        Returns:
            ValidationResult with findings.
        
        Raises:
            ValidationError: In fail_fast mode when error found.
        """
        all_findings: list[Finding] = []
        
        for validator in self.validators:
            findings = validator.validate_episode(episode, spec)
            all_findings.extend(findings)
            
            # Check for fail-fast
            if self.mode == ExecutionMode.FAIL_FAST:
                has_error = any(f.severity == Severity.ERROR for f in findings)
                has_warn = any(f.severity == Severity.WARN for f in findings)
                
                if has_error or (self.fail_on_warn and has_warn):
                    self.report.add_episode_result(all_findings)
                    raise ValidationError(
                        f"Validation failed for {episode.episode_id}",
                        findings=all_findings,
                    )
        
        # Update report
        self.report.add_episode_result(all_findings)
        
        # Determine validity
        has_error = any(f.severity == Severity.ERROR for f in all_findings)
        has_warn = any(f.severity == Severity.WARN for f in all_findings)
        is_valid = not has_error and not has_warn
        is_quarantined = has_error or (self.mode == ExecutionMode.QUARANTINE and has_warn)
        
        return ValidationResult(
            episode_id=episode.episode_id,
            findings=all_findings,
            is_valid=is_valid,
            is_quarantined=is_quarantined,
        )
    
    def validate_episodes(
        self, episodes: Iterator[Episode], spec: DatasetSpec
    ) -> Iterator[tuple[Episode, ValidationResult]]:
        """Validate multiple episodes.
        
        Args:
            episodes: Iterator of episodes.
            spec: Dataset specification.
        
        Yields:
            Tuples of (episode, result).
        """
        for episode in episodes:
            result = self.validate_episode(episode, spec)
            yield episode, result


class ValidationError(Exception):
    """Raised when validation fails in fail_fast mode."""
    
    def __init__(self, message: str, findings: list[Finding] | None = None) -> None:
        super().__init__(message)
        self.findings = findings or []


class ReportWriter:
    """Write validation reports to files."""
    
    def __init__(self, output_dir: Path | str) -> None:
        """Initialize report writer.
        
        Args:
            output_dir: Directory for report files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def summary_path(self) -> Path:
        """Path to summary JSON."""
        return self.output_dir / "validation_summary.json"
    
    @property
    def findings_path(self) -> Path:
        """Path to findings JSONL."""
        return self.output_dir / "findings.jsonl"
    
    def write_report(self, report: ValidationReport) -> None:
        """Write complete validation report.
        
        Args:
            report: ValidationReport to write.
        """
        self.write_summary(report)
        self.write_findings(report.findings)
    
    def write_summary(self, report: ValidationReport) -> None:
        """Write summary JSON."""
        with open(self.summary_path, "w") as f:
            json.dump(report.summary(), f, indent=2)
    
    def write_findings(self, findings: list[Finding]) -> None:
        """Write findings JSONL."""
        with open(self.findings_path, "w") as f:
            for finding in findings:
                f.write(json.dumps(finding.to_dict()) + "\n")
    
    def append_finding(self, finding: Finding) -> None:
        """Append a single finding to JSONL."""
        with open(self.findings_path, "a") as f:
            f.write(json.dumps(finding.to_dict()) + "\n")
