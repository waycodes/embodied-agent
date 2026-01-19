"""CLI entrypoint for EmbodiedDataKit."""

import click
from rich.console import Console

from embodied_datakit import __version__

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="edk")
@click.option("-v", "--verbose", count=True, help="Increase verbosity")
@click.option("-q", "--quiet", is_flag=True, help="Suppress non-error output")
@click.option(
    "--log-format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Log output format",
)
@click.option("--log-file", type=click.Path(), help="Write logs to file")
@click.option("-c", "--config", type=click.Path(exists=True), help="Config file path")
@click.pass_context
def main(
    ctx: click.Context,
    verbose: int,
    quiet: bool,
    log_format: str,
    log_file: str | None,
    config: str | None,
) -> None:
    """EmbodiedDataKit - Dataset compiler for robot trajectories."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["log_format"] = log_format
    ctx.obj["log_file"] = log_file
    ctx.obj["config"] = config


@main.command()
@click.argument("source")
@click.option("--split", default="train", help="Split to probe")
@click.option("-n", "--sample", default=0, help="Number of episodes to sample")
@click.option("-o", "--output", type=click.Path(), help="Output path")
@click.option("--format", "fmt", type=click.Choice(["yaml", "json"]), default="yaml")
@click.pass_context
def ingest(
    ctx: click.Context,
    source: str,
    split: str,
    sample: int,
    output: str | None,
    fmt: str,
) -> None:
    """Probe a source dataset and optionally sample episodes."""
    console.print(f"[bold]Probing:[/bold] {source}")
    console.print(f"  Split: {split}")
    console.print(f"  Sample: {sample} episodes")
    # TODO: Implement ingestion logic
    console.print("[yellow]Not yet implemented[/yellow]")


@main.command()
@click.argument("dataset")
@click.option("--split", default="train", help="Split to validate")
@click.option("--slice", "slice_", default=None, help="Slice selector")
@click.option("--max-episodes", type=int, help="Maximum episodes to validate")
@click.option("-r", "--report", type=click.Path(), help="Report output path")
@click.option("--format", "fmt", type=click.Choice(["json", "html", "csv"]), default="json")
@click.option("--fail-on-warn", is_flag=True, help="Exit with error on warnings")
@click.option("--strict", is_flag=True, help="Enable all optional validations")
@click.pass_context
def validate(
    ctx: click.Context,
    dataset: str,
    split: str,
    slice_: str | None,
    max_episodes: int | None,
    report: str | None,
    fmt: str,
    fail_on_warn: bool,
    strict: bool,
) -> None:
    """Run validation on a dataset."""
    console.print(f"[bold]Validating:[/bold] {dataset}")
    # TODO: Implement validation logic
    console.print("[yellow]Not yet implemented[/yellow]")


@main.command()
@click.argument("source")
@click.option("-o", "--output", required=True, type=click.Path(), help="Output directory")
@click.option("--split", default="train", help="Split to compile")
@click.option("--slice", "slice_", default=None, help="Slice selector")
@click.option("--pipeline", type=click.Path(exists=True), help="Pipeline config path")
@click.option("--camera", default=None, help="Canonical camera name")
@click.option("--resolution", default="256x256", help="Image resize resolution")
@click.option(
    "--action-mapping",
    type=click.Choice(["passthrough", "ee7", "config"]),
    default="passthrough",
)
@click.option("--normalize-actions", is_flag=True, help="Apply action normalization")
@click.option("--skip-validation", is_flag=True, help="Skip validation")
@click.option("--episodes-per-shard", type=int, default=1000)
@click.option("--max-video-frames", type=int, default=10000)
@click.option("--video-crf", type=int, default=23)
@click.option("-j", "--workers", type=int, default=1, help="Parallel workers")
@click.option("--fail-fast", is_flag=True)
@click.option("--quarantine", is_flag=True)
@click.option("--resume", is_flag=True)
@click.option("--seed", type=int, default=42)
@click.pass_context
def compile(
    ctx: click.Context,
    source: str,
    output: str,
    split: str,
    slice_: str | None,
    pipeline: str | None,
    camera: str | None,
    resolution: str,
    action_mapping: str,
    normalize_actions: bool,
    skip_validation: bool,
    episodes_per_shard: int,
    max_video_frames: int,
    video_crf: int,
    workers: int,
    fail_fast: bool,
    quarantine: bool,
    resume: bool,
    seed: int,
) -> None:
    """Compile a source dataset to LeRobotDataset v3 format."""
    console.print(f"[bold]Compiling:[/bold] {source}")
    console.print(f"  Output: {output}")
    console.print(f"  Split: {split}")
    # TODO: Implement compilation logic
    console.print("[yellow]Not yet implemented[/yellow]")


@main.command()
@click.argument("dataset", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output path for index")
@click.option("--rebuild", is_flag=True, help="Force rebuild")
@click.pass_context
def index(ctx: click.Context, dataset: str, output: str | None, rebuild: bool) -> None:
    """Build or rebuild the episode index."""
    console.print(f"[bold]Indexing:[/bold] {dataset}")
    # TODO: Implement indexing logic
    console.print("[yellow]Not yet implemented[/yellow]")


@main.command()
@click.argument("dataset", type=click.Path(exists=True))
@click.option("-q", "--query", required=True, help="Query predicate")
@click.option("-o", "--output", required=True, type=click.Path(), help="Output path")
@click.option("--mode", type=click.Choice(["copy", "view"]), default="copy")
@click.option("--limit", type=int, help="Maximum episodes")
@click.pass_context
def slice(
    ctx: click.Context,
    dataset: str,
    query: str,
    output: str,
    mode: str,
    limit: int | None,
) -> None:
    """Create a dataset subset based on query predicates."""
    console.print(f"[bold]Slicing:[/bold] {dataset}")
    console.print(f"  Query: {query}")
    # TODO: Implement slicing logic
    console.print("[yellow]Not yet implemented[/yellow]")


@main.command("export-rlds")
@click.argument("dataset", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, type=click.Path(), help="Output directory")
@click.option("--name", default=None, help="TFDS dataset name")
@click.option("--episodes-per-file", type=int, default=100)
@click.option("--include-video/--no-include-video", default=True)
@click.pass_context
def export_rlds(
    ctx: click.Context,
    dataset: str,
    output: str,
    name: str | None,
    episodes_per_file: int,
    include_video: bool,
) -> None:
    """Export compiled dataset to RLDS/TFDS format."""
    console.print(f"[bold]Exporting to RLDS:[/bold] {dataset}")
    console.print(f"  Output: {output}")
    # TODO: Implement RLDS export logic
    console.print("[yellow]Not yet implemented[/yellow]")


@main.command()
@click.argument("dataset")
@click.option("--split", default="train", help="Split to inspect")
@click.option("--show-samples", type=int, default=3)
@click.option("--format", "fmt", type=click.Choice(["text", "json", "markdown"]), default="text")
@click.pass_context
def inspect(ctx: click.Context, dataset: str, split: str, show_samples: int, fmt: str) -> None:
    """Inspect schema and samples from a dataset."""
    console.print(f"[bold]Inspecting:[/bold] {dataset}")
    # TODO: Implement inspection logic
    console.print("[yellow]Not yet implemented[/yellow]")


if __name__ == "__main__":
    main()
