"""Integration tests for compile→write pipeline."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from tests.fixtures import generate_synthetic_dataset


class TestCompileWritePipeline:
    """Integration tests for the compile→write pipeline."""

    def test_compile_to_lerobot_v3(self) -> None:
        """Test compiling synthetic data to LeRobot v3 format."""
        from embodied_datakit.compiler import Compiler
        from embodied_datakit.validators import RLDSInvariantValidator
        from embodied_datakit.writers import LeRobotV3Writer

        # Generate synthetic data
        episodes, spec = generate_synthetic_dataset(num_episodes=3, steps_per_episode=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create compiler with writer and validator
            compiler = Compiler()
            compiler.add_validator(RLDSInvariantValidator())
            compiler.set_writer(LeRobotV3Writer())

            # Manually compile (since we have episodes directly)
            writer = LeRobotV3Writer()
            writer.begin(spec, output_dir)

            for episode in episodes:
                writer.write_episode(episode)

            artifacts = writer.finalize()

            # Verify output files exist
            assert (output_dir / "meta" / "info.json").exists()
            assert (output_dir / "meta" / "tasks.jsonl").exists()
            assert (output_dir / "meta" / "stats.json").exists()
            assert (output_dir / "meta" / "episodes" / "episodes.parquet").exists()

            # Verify info.json content
            with open(output_dir / "meta" / "info.json") as f:
                info = json.load(f)
            assert info["total_episodes"] == 3
            assert info["total_frames"] == 15  # 3 episodes * 5 steps

            # Verify episodes parquet
            episodes_table = pq.read_table(output_dir / "meta" / "episodes" / "episodes.parquet")
            assert len(episodes_table) == 3

            # Verify steps parquet exists
            data_files = list((output_dir / "data").rglob("*.parquet"))
            assert len(data_files) > 0

            # Read and verify steps
            steps_table = pq.read_table(data_files[0])
            assert "episode_index" in steps_table.column_names
            assert "frame_index" in steps_table.column_names

    def test_validation_during_compile(self) -> None:
        """Test that validation runs during compilation."""
        from embodied_datakit.validators import RLDSInvariantValidator, ValidationReport

        episodes, spec = generate_synthetic_dataset(num_episodes=2, steps_per_episode=3)

        validator = RLDSInvariantValidator()
        report = ValidationReport()

        for episode in episodes:
            findings = validator.validate_episode(episode, spec)
            report.add_episode_result(findings)

        # Synthetic episodes should be valid
        assert report.total_episodes == 2
        assert report.error_episodes == 0

    def test_stats_computation(self) -> None:
        """Test that statistics are computed correctly."""
        import tempfile
        from pathlib import Path

        from embodied_datakit.writers import LeRobotV3Writer

        episodes, spec = generate_synthetic_dataset(num_episodes=5, steps_per_episode=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            writer = LeRobotV3Writer()
            writer.begin(spec, output_dir)

            for episode in episodes:
                writer.write_episode(episode)

            writer.finalize()

            # Verify stats.json
            with open(output_dir / "meta" / "stats.json") as f:
                stats = json.load(f)

            # Should have action stats
            assert "action" in stats
            assert "mean" in stats["action"]
            assert "std" in stats["action"]
            assert len(stats["action"]["mean"]) == 7  # action_dim


class TestTransformPipeline:
    """Integration tests for transform pipeline."""

    def test_camera_selection_transform(self) -> None:
        """Test camera selection in pipeline."""
        from embodied_datakit.transforms import SelectCameraTransform

        episodes, spec = generate_synthetic_dataset(num_episodes=1, steps_per_episode=3)
        episode = episodes[0]

        transform = SelectCameraTransform(camera_name="front")
        transformed = transform.transform_episode(episode, spec)

        # Should have canonical camera key
        assert "observation.images.canonical" in transformed.steps[0].observation

    def test_action_normalization_transform(self) -> None:
        """Test action normalization in pipeline."""
        from embodied_datakit.transforms import NormalizeActionsTransform

        episodes, spec = generate_synthetic_dataset(num_episodes=1, steps_per_episode=5)
        episode = episodes[0]

        transform = NormalizeActionsTransform(action_bounds=(-1.0, 1.0))
        transformed = transform.transform_episode(episode, spec)

        # Actions should be normalized
        for step in transformed.steps:
            if step.action is not None:
                assert np.all(step.action >= -1.0)
                assert np.all(step.action <= 1.0)


class TestValidationPipeline:
    """Integration tests for validation pipeline."""

    def test_full_validation_suite(self) -> None:
        """Test running all validators on synthetic data."""
        from embodied_datakit.validators import (
            ActionSanityValidator,
            EpisodeLengthValidator,
            RLDSInvariantValidator,
            SchemaValidator,
            TimestampValidator,
        )

        episodes, spec = generate_synthetic_dataset(num_episodes=3, steps_per_episode=10)

        validators = [
            RLDSInvariantValidator(),
            EpisodeLengthValidator(min_length=5, max_length=100),
            TimestampValidator(),
            ActionSanityValidator(),
            SchemaValidator(),
        ]

        total_findings = []
        for episode in episodes:
            for validator in validators:
                findings = validator.validate_episode(episode, spec)
                total_findings.extend(findings)

        # Synthetic data should pass all validators
        errors = [f for f in total_findings if f.severity.value == "ERROR"]
        assert len(errors) == 0, f"Unexpected errors: {[f.message for f in errors]}"


class TestRoundTrip:
    """Round-trip integration tests for write then read."""

    def test_write_read_episodes_parquet(self) -> None:
        """Test writing and reading episodes metadata."""
        from embodied_datakit.artifacts import ArtifactLayout
        from embodied_datakit.writers import EpisodesTableWriter

        episodes, spec = generate_synthetic_dataset(num_episodes=3, steps_per_episode=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            layout = ArtifactLayout(tmpdir)
            layout.create_dirs()

            writer = EpisodesTableWriter(layout.episodes_index_path)

            for i, episode in enumerate(episodes):
                writer.add_episode(
                    episode=episode,
                    spec=spec,
                    parquet_file=f"chunk-000/episode_{i:06d}.parquet",
                    parquet_row_start=i * 5,
                    parquet_row_end=(i + 1) * 5,
                )

            writer.write()

            # Read back and verify
            table = pq.read_table(layout.episodes_index_path)
            assert len(table) == 3

            # Verify fields
            assert "episode_id" in table.column_names
            assert "num_steps" in table.column_names
            assert "parquet_row_start" in table.column_names

            # Verify values
            rows = table.to_pydict()
            assert rows["num_steps"] == [5, 5, 5]
            assert rows["parquet_row_start"] == [0, 5, 10]

    def test_artifact_layout_structure(self) -> None:
        """Test artifact layout creates correct structure."""
        from embodied_datakit.artifacts import ArtifactLayout

        with tempfile.TemporaryDirectory() as tmpdir:
            layout = ArtifactLayout(tmpdir)
            layout.create_dirs()

            # Verify directories exist
            assert layout.meta_dir.exists()
            assert layout.data_dir.exists()
            assert layout.videos_dir.exists()
            assert layout.reports_dir.exists()
            assert layout.logs_dir.exists()

            # Verify path methods
            assert layout.info_path == layout.meta_dir / "info.json"
            assert layout.tasks_path == layout.meta_dir / "tasks.jsonl"


class TestIngestionSmoke:
    """Smoke tests for ingestion pipeline."""

    def test_single_episode_ingestion(self) -> None:
        """Test ingesting a single episode from synthetic data."""
        episodes, spec = generate_synthetic_dataset(num_episodes=1, steps_per_episode=10)
        episode = episodes[0]

        # Verify episode structure
        assert episode.episode_id is not None
        assert len(episode.steps) == 10
        assert episode.steps[0].is_first
        assert episode.steps[-1].is_last

        # Verify step iteration
        step_count = 0
        for step in episode.iter_steps():
            assert step.observation is not None
            step_count += 1
        assert step_count == 10

    def test_streaming_safety(self) -> None:
        """Test that episodes can be processed in streaming fashion."""
        episodes, spec = generate_synthetic_dataset(num_episodes=5, steps_per_episode=10)

        # Simulate streaming processing
        processed_ids = []
        for episode in episodes:
            # Process episode
            assert episode.num_steps > 0
            assert episode.episode_id not in processed_ids
            processed_ids.append(episode.episode_id)

            # Verify we can access all steps
            for step in episode.steps:
                assert step.observation is not None

        assert len(processed_ids) == 5

    def test_episode_observations_non_empty(self) -> None:
        """Test that observations are non-empty."""
        episodes, spec = generate_synthetic_dataset(num_episodes=1, steps_per_episode=5)
        episode = episodes[0]

        for step in episode.steps:
            # Should have at least one observation key
            assert len(step.observation) > 0

            # Check image observations exist
            image_keys = [k for k in step.observation if k.startswith("observation.images.")]
            assert len(image_keys) > 0

            # Check images are valid arrays
            for key in image_keys:
                img = step.observation[key]
                assert isinstance(img, np.ndarray)
                assert img.ndim == 3  # H, W, C
                assert img.shape[2] == 3  # RGB

    def test_action_validity(self) -> None:
        """Test that actions are valid for non-terminal steps."""
        episodes, spec = generate_synthetic_dataset(num_episodes=1, steps_per_episode=5)
        episode = episodes[0]

        for i, step in enumerate(episode.steps):
            if not step.is_last:
                # Non-last steps should have valid actions
                assert step.action is not None
                assert isinstance(step.action, np.ndarray)
                assert step.action.ndim == 1
