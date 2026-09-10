"""
CLI entry point for the pediatric counting pipeline.

Usage:
    pediatric-counter run --config configs/room_default.yaml
    pediatric-counter run --config configs/room_default.yaml --video path/to/video.mp4
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import click
from rich.console import Console

console = Console(highlight=False)

# ---------------------------------------------------------------------------
# Epilog shown at the bottom of --help
# ---------------------------------------------------------------------------
_EPILOG = """\b
─────────────────────────────────────────────────────
  QUICK EXAMPLES
─────────────────────────────────────────────────────

  1. Just run the wizard (easiest — no flags needed):
       python run.py

  2. Analyse a specific video file:
       python run.py --video videos/opd_feed1.mp4

  3. Skip the first 2 minutes (2 min = 3600 frames at 30fps):
       python run.py --video videos/opd_feed1.mp4 -s 3600

  4. Watch detections live on screen while it runs:
       python run.py --video videos/opd_feed1.mp4 --live

  5. Speed up the live view to 2x:
       python run.py --video videos/opd_feed1.mp4 --live --speed 2.0

  6. Count adults AND children separately:
       python run.py --video videos/opd_feed1.mp4 --count-adults

  7. Quick test — only process the first 500 frames:
       python run.py --video videos/opd_feed1.mp4 -m 500

  8. Read the full user manual in the terminal:
       python run.py docs

─────────────────────────────────────────────────────
  TIP  Not sure what to type? Just run:  python run.py
       The system will guide you with a step-by-step menu.
─────────────────────────────────────────────────────
"""


def cli_entrypoint() -> None:
    """Entry point that seamlessly routes direct CLI options (e.g. -s 1450 --live) into the run command."""
    if len(sys.argv) > 1 and sys.argv[1] not in ("run", "docs", "evaluate", "--help", "-h"):
        sys.argv.insert(1, "run")
    main()


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """Pediatric & Adult CCTV Counting System.\n
    \b
    Run without arguments to launch the interactive step-by-step wizard.
    Use --help to see all available options.
    Use 'python run.py docs' to read the full user manual in the terminal.
    """
    if ctx.invoked_subcommand is None:
        # Launch interactive wizard
        from run import main as run_wizard
        run_wizard()


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------

_common_options = [
    click.option(
        "--config", "-c",
        default=Path("pediatric_counter/configs/room_default.yaml"),
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="Path to YAML room config file. [default: configs/room_default.yaml]",
    ),
    click.option(
        "--video", "-v",
        default=None,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="Path to the video file you want to analyse.",
    ),
    click.option(
        "--classifier-path",
        default=None,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="Path to Stage 2 demographic classifier (.pt or .onnx).",
    ),
    click.option(
        "--detector-path",
        default=None,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="Path to Stage 1 person detector model (.pt).",
    ),
    click.option(
        "--out", "-o",
        default=None,
        type=click.Path(file_okay=False, path_type=Path),
        help="Folder where results (counts, timeline, exports) will be saved.",
    ),
    click.option(
        "--start-frame", "-s",
        default=None,
        type=int,
        help="Frame number to start from. Skip an intro or jump to a busy period.",
    ),
    click.option(
        "--max-frames", "-m",
        default=None,
        type=int,
        help="Stop after this many frames.",
    ),
    click.option(
        "--live/--no-live",
        default=None,
        help="Show a live detection window while analysing. Controls: SPACE=pause Q=quit +/-=speed",
    ),
    click.option(
        "--speed",
        default=None,
        type=float,
        help="Live window playback speed (1.0=real speed, 2.0=2x speed, 0=max).",
    ),
    click.option(
        "--enable-pose/--no-enable-pose",
        default=None,
        help="Enable skeletal pose estimation & cephalocaudal ratio verification plugin.",
    ),
    click.option(
        "--count-adults/--no-count-adults",
        default=None,
        help="Also count adults separately alongside children.",
    ),
]


def add_common_options(func):
    for opt in reversed(_common_options):
        func = opt(func)
    return func


def _execute_pipeline(
    config: Path,
    video: Path | None,
    classifier_path: Path | None,
    detector_path: Path | None,
    out: Path | None,
    start_frame: int | None,
    max_frames: int | None,
    live: bool | None,
    speed: float | None,
    enable_pose: bool | None,
    count_adults: bool | None,
) -> None:
    from pediatric_counter.app.config import PipelineConfig

    console.rule("[bold cyan]Pediatric Counting -- Pipeline Start")
    cfg = PipelineConfig.from_yaml(config)

    if video:
        cfg.room.video_path = video
    if classifier_path:
        cfg.models.classifier_path = classifier_path
    if detector_path:
        cfg.models.detector_path = detector_path
    if out:
        cfg.artifacts.output_dir = out
    if start_frame is not None:
        cfg.room.start_frame = start_frame
    if max_frames is not None:
        cfg.room.max_frames = max_frames
    if live is not None:
        cfg.artifacts.live_view = live
    if speed is not None:
        cfg.artifacts.playback_speed = speed
    if enable_pose is not None:
        cfg.pose.enabled = enable_pose
    if count_adults is not None:
        cfg.counting.count_adults = count_adults

    console.print(f"[green]Room:[/green]       {cfg.room.id}")
    console.print(f"[green]Video:[/green]      {cfg.room.video_path}")
    console.print(f"[green]Detector:[/green]   {cfg.models.detector_path}")
    console.print(f"[green]Classifier:[/green] {cfg.models.classifier_path}")
    if cfg.pose.enabled:
        console.print(f"[green]Pose Plugin:[/green] Active ({cfg.pose.model_path})")
    console.print(f"[green]Tracker:[/green]    {cfg.tracking.backend}")
    console.print(f"[green]Output:[/green]     {cfg.artifacts.output_dir}")
    console.print(f"[dim]Config hash:    {cfg.config_hash()}[/dim]")

    # Lazy import so CLI stays fast even if torch not installed
    from pediatric_counter.app.pipeline import run_pipeline
    run_pipeline(cfg)


@main.command(name="run", epilog=_EPILOG)
@add_common_options
def run(**kwargs) -> None:
    """Analyse a video and count children (and optionally adults).\n
    \b
    Tip: not sure which options to use? Just run: python run.py
    and follow the interactive prompts instead.
    """
    _execute_pipeline(**kwargs)


@main.command(name="count", epilog=_EPILOG)
@add_common_options
def count_cmd(**kwargs) -> None:
    """Alias for run: analyse a video and count children and adults."""
    _execute_pipeline(**kwargs)


# ---------------------------------------------------------------------------
# docs / man-page command
# ---------------------------------------------------------------------------

@main.command(name="docs")
def docs_command() -> None:
    """Read the full user manual right here in the terminal (like 'man' on Linux)."""
    manual_path = Path(__file__).parents[2] / "USER_MANUAL.md"

    if not manual_path.exists():
        console.print(
            "[bold red]Error:[/bold red] USER_MANUAL.md not found at "
            f"{manual_path}.  Please ensure the file exists in the project root."
        )
        raise SystemExit(1)

    try:
        from rich.markdown import Markdown
        md_text = manual_path.read_text(encoding="utf-8")
        console.print()
        console.rule("[bold cyan]\U0001F4D6  Pediatric Counter — User Manual")
        console.print()
        console.print(Markdown(md_text))
        console.print()
        console.rule("[dim]End of manual  •  Run 'python run.py --help' for a quick option reference[/dim]")
    except Exception as exc:  # pragma: no cover
        console.print(f"[bold red]Could not render manual:[/bold red] {exc}")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# evaluate command
# ---------------------------------------------------------------------------

@main.command(name="evaluate")
@click.option(
    "--frames", "-m",
    default=120,
    type=int,
    help="Number of frames to process per video during evaluation (default: 120).",
)
@click.option(
    "--compare/--no-compare",
    default=False,
    help="Run comparative sweep comparing PyTorch .pt vs ONNX Runtime .onnx across all feeds.",
)
def evaluate_command(frames: int, compare: bool) -> None:
    """Run full evaluation suite across all available CCTV videos."""
    if compare:
        from pediatric_counter.evaluation.benchmark_suite import run_comparative_sweep
        run_comparative_sweep(max_frames_per_video=frames)
    else:
        from pediatric_counter.evaluation.benchmark_suite import run_all_evaluations
        run_all_evaluations(max_frames_per_video=frames)


if __name__ == "__main__":
    main()

