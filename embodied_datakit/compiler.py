"""Compiler - Pipeline orchestrator for dataset compilation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from rich.progress import Progress, TaskID

from embodied_datakit.adapters.base import Adapter
from embodied_datakit.config import Config
from embodied_datakit.logging import EDKLogger, get_logger
from embodied_datakit.random import set_seed
from embodied_datakit.schema.episode import Episode
from embodied_datakit.schema.spec import DatasetSpec
from embodied_datakit.transforms.base import Transform, TransformChain
from embodied_datakit.validators.base import (
    Finding,
    Severity,
    ValidationReport,
    Validator,
)
from embodied_datakit.writers.base import Writer


@dataclass
class CompileResult:
    """Result of a compilation run.

    Attributes:
        success: Whether compilation succeeded.
        output_dir: Path to output directory.
        spec: Dataset specification.
        validation_report: Validation results.
        artifacts: List of produced artifact paths.
        episodes_written: Number of episodes successfully written.
        episodes_rejected: Number of episodes rejected due to errors.
        episodes_invalid: Number of episodes marked invalid.
    """

    success: bool
    output_dir: Path
    spec: DatasetSpec
    validation_report: ValidationReport
    artifacts: list[Path] = field(default_factory=list)
    episodes_written: int = 0
    episodes_rejected: int = 0
    episodes_invalid: int = 0


class Compiler:
    """Pipeline orchestrator for dataset compilation.

    Orchestrates: Adapter → Transform chain → Validate → Write → Index.
    """

    def __init__(
        self,
        config: Config | None = None,
        logger: EDKLogger | None = None,
    ) -> None:
        """Initialize compiler.

        Args:
            config: Compilation configuration.
            logger: Logger instance.
        """
        self.config = config or Config()
        self.logger = logger or get_logger()

        # Pipeline components
        self.transforms: TransformChain = TransformChain()
        self.validators: list[Validator] = []
        self.writer: Writer | None = None

    def add_transform(self, transform: Transform) -> "Compiler":
        """Add a transform to the pipeline.

        Args:
            transform: Transform to add.

        Returns:
            Self for method chaining.
        """
        self.transforms.add(transform)
        return self

    def add_validator(self, validator: Validator) -> "Compiler":
        """Add a validator to the pipeline.

        Args:
            validator: Validator to add.

        Returns:
            Self for method chaining.
        """
        self.validators.append(validator)
        return self

    def set_writer(self, writer: Writer) -> "Compiler":
        """Set the output writer.

        Args:
            writer: Writer instance.

        Returns:
            Self for method chaining.
        """
        self.writer = writer
        return self

    def compile(
        self,
        adapter: Adapter,
        output_dir: Path | str,
        split: str = "train",
        selector: str | None = None,
    ) -> CompileResult:
        """Compile a dataset.

        Args:
            adapter: Source adapter.
            output_dir: Output directory path.
            split: Split to compile.
            selector: Optional slice selector.

        Returns:
            Compilation result.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed
        set_seed(self.config.seed)

        # Probe dataset
        self.logger.info(f"Probing dataset...")
        spec = adapter.probe()
        spec.transform_pipeline = self.transforms.names

        # Initialize writer
        if self.writer is None:
            raise ValueError("No writer configured")
        self.writer.begin(spec, output_dir)

        # Initialize tracking
        report = ValidationReport()
        episodes_written = 0
        episodes_rejected = 0
        episodes_invalid = 0

        # Process episodes
        self.logger.info(f"Compiling split '{split}' with selector '{selector}'...")

        with Progress() as progress:
            task = progress.add_task("Processing episodes...", total=None)

            for episode in adapter.iter_episodes(split, selector):
                progress.advance(task)

                # Transform
                episode = self._apply_transforms(episode, spec)

                # Validate
                findings = self._validate_episode(episode, spec)
                report.add_episode_result(findings)

                # Check for errors
                has_error = any(f.severity == Severity.ERROR for f in findings)
                has_warn = any(f.severity == Severity.WARN for f in findings)

                if has_error:
                    episodes_rejected += 1
                    if self.config.validation.fail_fast:
                        self.logger.error(f"Episode {episode.episode_id} rejected, aborting")
                        break
                    continue

                # Mark invalid if warnings
                if has_warn:
                    episode.invalid = True
                    episodes_invalid += 1

                # Write episode
                self.writer.write_episode(episode)
                episodes_written += 1

        # Finalize
        self.logger.info("Finalizing output...")
        artifacts = self.writer.finalize()

        # Close adapter
        adapter.close()

        success = not report.has_errors() or episodes_written > 0

        self.logger.info(
            f"Compilation complete: {episodes_written} written, "
            f"{episodes_rejected} rejected, {episodes_invalid} invalid"
        )

        return CompileResult(
            success=success,
            output_dir=output_dir,
            spec=spec,
            validation_report=report,
            artifacts=artifacts,
            episodes_written=episodes_written,
            episodes_rejected=episodes_rejected,
            episodes_invalid=episodes_invalid,
        )

    def _apply_transforms(self, episode: Episode, spec: DatasetSpec) -> Episode:
        """Apply transform chain to episode."""
        return self.transforms.transform_episode(episode, spec)

    def _validate_episode(self, episode: Episode, spec: DatasetSpec) -> list[Finding]:
        """Run all validators on an episode."""
        findings: list[Finding] = []
        for validator in self.validators:
            findings.extend(validator.validate_episode(episode, spec))
        return findings

    def validate_only(
        self,
        adapter: Adapter,
        split: str = "train",
        selector: str | None = None,
        max_episodes: int | None = None,
    ) -> ValidationReport:
        """Run validation without writing output.

        Args:
            adapter: Source adapter.
            split: Split to validate.
            selector: Optional slice selector.
            max_episodes: Maximum episodes to validate.

        Returns:
            Validation report.
        """
        spec = adapter.probe()
        report = ValidationReport()

        count = 0
        for episode in adapter.iter_episodes(split, selector):
            # Transform
            episode = self._apply_transforms(episode, spec)

            # Validate
            findings = self._validate_episode(episode, spec)
            report.add_episode_result(findings)

            count += 1
            if max_episodes and count >= max_episodes:
                break

        adapter.close()
        return report
