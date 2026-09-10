"""
Interactive Menu Launcher for Pediatric & Adult CCTV Counter.

Usage:
    python run.py
or double-click start.bat
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from pediatric_counter.app.config import PipelineConfig
from pediatric_counter.app.pipeline import run_pipeline
from pediatric_counter.counting.events import parse_cctv_start_time

console = Console(highlight=False)


def scan_cctv_videos() -> list[dict]:
    """Scan assets/cctv_videos and separate playable videos from empty placeholders with active section hints."""
    videos_dir = Path("assets/cctv_videos")
    if not videos_dir.exists():
        return []

    # Map known CCTV shift patterns to their optimal active movement sections
    active_hints = {
        "20260616055958": (1450, "Hallway Opening & Morning Passersby (~00:58)"),
        "20260616095600": (430, "High-Activity OPD Patient Queue (40+ people, ~00:17)"),
        "20260616150730": (0, "Afternoon Clinic Intake & Entry (~00:00)"),
        "20260625175955": (0, "Evening Shift Entry Corridor (~00:00)"),
        "20260708075415": (0, "Hospital Pharmacy Lobby & Dispensary Counter (~00:00)"),
    }

    playable = []
    for f in sorted(videos_dir.glob("*.mp4")):
        size_mb = f.stat().st_size / (1024 * 1024)
        try:
            with f.open("rb") as fp:
                head = fp.read(16)
                is_valid = b"IMKH" in head or b"ftyp" in head
        except Exception:
            is_valid = False

        if not is_valid:
            continue  # Skip zero-filled download placeholders

        dt = parse_cctv_start_time(f)
        rec_frame = 0
        rec_desc = "Start of recording (~00:00)"

        if "cctv_child_room_drawers" in f.name:
            label = "👶 Child Activity Feed — Playroom & Drawers"
            rec_frame = 0
            rec_desc = "Active child toddler movement from frame 0"
            is_child_feed = True
        elif "cctv_kindergarten_classroom" in f.name:
            label = "👶 Kindergarten CCTV — Classroom Playgroup"
            rec_frame = 260
            rec_desc = "Toddler classroom playgroup (~00:08, frame 260)"
            is_child_feed = True
        elif "Pharmacy" in f.name:
            shift_date = dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "2026-07-08 07:54:15"
            label = f"🏥 Hospital Pharmacy Lobby — {shift_date} ({size_mb:.0f} MB)"
            rec_frame = 0
            rec_desc = "Pharmacy Waiting & Dispensary Counter (~00:00)"
            is_child_feed = False
        elif dt:
            shift_date = dt.strftime("%Y-%m-%d")
            shift_time = dt.strftime("%H:%M:%S")
            hour = dt.hour
            if hour < 9:
                shift_name = "Morning Shift"
            elif hour < 14:
                shift_name = "Midday Shift"
            elif hour < 18:
                shift_name = "Afternoon Shift"
            else:
                shift_name = "Evening Shift"
            label = f"{shift_date} {shift_time} - {shift_name} ({size_mb:.0f} MB)"
            is_child_feed = False

            # Check for specific shift active section
            time_key = dt.strftime("%Y%m%d%H%M%S")
            if time_key in active_hints:
                rec_frame, rec_desc = active_hints[time_key]
        else:
            label = f"{f.name[:45]} ({size_mb:.0f} MB)"
            is_child_feed = False

        playable.append({
            "path": f,
            "filename": f.name,
            "label": label,
            "size_mb": size_mb,
            "start_time": dt,
            "recommended_start_frame": rec_frame,
            "recommended_desc": rec_desc,
            "is_child_feed": is_child_feed,
        })

    # Display child feeds at the top for immediate verification
    playable.sort(key=lambda x: (not x["is_child_feed"], x["filename"]))
    return playable


import json

def show_recent_timestamps():
    """Display timestamps from the latest run using a clean colored table."""
    events_path = Path("runs/otmc_opd_baseline/detection_events.json")
    if not events_path.exists():
        return
    try:
        with events_path.open("r", encoding="utf-8") as f:
            events = json.load(f)
    except Exception:
        return

    if not events:
        return

    table = Table(
        title="⏱️ Last Run Detection Events (Timeline)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Event #", justify="center", style="cyan", width=8)
    table.add_column("Track ID", justify="center", style="white", width=9)
    table.add_column("Demographic", justify="center", width=13)
    table.add_column("Video Time", justify="center", style="green")
    table.add_column("CCTV Clock Time", justify="center", style="yellow")
    table.add_column("Frames", justify="center", style="dim")
    table.add_column("Duration", justify="right", style="cyan")
    table.add_column("Confidence", justify="right", style="magenta")

    for ev in events:
        demo = ev.get("demographic", "unknown").upper()
        if demo == "CHILD":
            demo_badge = "[bold green]👶 CHILD[/bold green]"
        elif demo == "ADULT":
            demo_badge = "[bold blue]🧑 ADULT[/bold blue]"
        else:
            demo_badge = "[dim]❓ UNKNOWN[/dim]"

        clock = f"{ev.get('wall_clock_start')} - {ev.get('wall_clock_end')}" if ev.get("wall_clock_start") else "Live Feed"
        conf = f"{ev.get('avg_confidence', 0):.0%}"

        table.add_row(
            f"#{ev.get('event_id')}",
            f"#{ev.get('track_id')}",
            demo_badge,
            f"{ev.get('start_time')} - {ev.get('end_time')}",
            clock,
            f"{ev.get('start_frame')}-{ev.get('end_frame')}",
            f"{ev.get('duration_seconds')}s",
            conf,
        )

    console.print(table)


def main():
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]🧒 Pediatric & Adult CCTV Counting System[/bold cyan]\n"
            "[dim]Explainable Single-Room CCTV Demographic Research Harness[/dim]",
            border_style="cyan",
        )
    )

    default_cfg_path = Path("pediatric_counter/configs/room_default.yaml")
    if default_cfg_path.exists():
        cfg = PipelineConfig.from_yaml(default_cfg_path)
    else:
        console.print("[red]Error: default config not found.[/red]")
        return

    # ── 1. Select Video Feed or Live Camera ────────────────────────────────────
    videos = scan_cctv_videos()
    from pediatric_counter.io.camera import detect_available_cameras
    available_cameras = detect_available_cameras(max_to_test=2)

    console.print("\n[bold yellow]Step 1: Select Input Feed (Recorded Video or Live Camera)[/bold yellow]")
    table = Table(title="Available CCTV Feeds & Camera Sources", show_header=True, header_style="bold magenta")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Source Type & Shift", style="green")
    table.add_column("Details", style="dim", max_width=45)
    table.add_column("Size / Status", justify="right", style="yellow")

    feed_idx = 1
    for v in videos:
        fn = v["filename"]
        fn_display = fn[:40] + "..." if len(fn) > 42 else fn
        table.add_row(str(feed_idx), v["label"], fn_display, f"{v['size_mb']:.1f} MB")
        feed_idx += 1

    # Add auto-detected USB webcams
    cam_start_idx = feed_idx
    cam_map = {}
    for cam in available_cameras:
        table.add_row(
            str(feed_idx),
            f"[bold green]📷 {cam['label']}[/bold green]",
            "Plug-and-play USB camera (Zero-Latency Threaded)",
            "[bold green]CONNECTED[/bold green]",
        )
        cam_map[feed_idx] = cam["index"]
        feed_idx += 1

    # RTSP / Custom live stream option
    rtsp_option_idx = feed_idx
    table.add_row(
        str(rtsp_option_idx),
        "[bold cyan]🌐 Custom RTSP / Network IP Camera Stream[/bold cyan]",
        "Enter custom RTSP / HTTP URL for hospital CCTV network",
        "[bold cyan]RTSP STREAM[/bold cyan]",
    )
    console.print(table)

    choice = IntPrompt.ask(
        "Choose feed number",
        default=1,
        choices=[str(i) for i in range(1, rtsp_option_idx + 1)],
    )

    if choice in cam_map:
        cam_idx = cam_map[choice]
        cfg.room.video_path = Path(f"camera:{cam_idx}")
        chosen_label = f"Live USB Camera #{cam_idx}"
        active_frame = 0
        active_desc = "Live Real-Time Camera Stream"
    elif choice == rtsp_option_idx:
        console.print("\n[bold cyan]RTSP Network Stream Configuration:[/bold cyan]")
        rtsp_url = Prompt.ask("Enter RTSP Stream URL (e.g. rtsp://192.168.1.100:554/stream1)", default="rtsp://localhost:554/live")
        cfg.room.video_path = Path(f"camera:{rtsp_url}")
        chosen_label = f"RTSP Stream ({rtsp_url[:30]}...)"
        active_frame = 0
        active_desc = "Network CCTV RTSP Stream"
    else:
        chosen_video = videos[choice - 1]
        cfg.room.video_path = chosen_video["path"]
        chosen_label = chosen_video["label"]
        active_frame = chosen_video.get("recommended_start_frame", 0)
        active_desc = chosen_video.get("recommended_desc", "Active section")
        if chosen_video.get("is_child_feed", False):
            cfg.room.demographic_prior = "child_dominant"
        else:
            cfg.room.demographic_prior = "adult_dominant"

    # ── 2. Select Run Mode ─────────────────────────────────────────────────────
    console.print("\n[bold yellow]Step 2: Select Run / Analysis Mode[/bold yellow]")
    console.print(f"  [cyan][1][/cyan] [bold green]Recommended Active Section[/bold green] ({active_desc} - fast verification)")
    console.print("  [cyan][2][/cyan] [blue]Quick 10-Second Test[/blue] (Process first 250 frames from 00:00)")
    console.print("  [cyan][3][/cyan] [magenta]Jump to Specific Timestamp[/magenta] (Enter elapsed minute:second or frame number)")
    console.print("  [cyan][4][/cyan] [yellow]Complete Shift Run[/yellow] (Process entire hospital shift to final report)")

    mode_choice = IntPrompt.ask("Choose mode", default=1, choices=["1", "2", "3", "4"])

    if mode_choice == 1:
        cfg.room.start_frame = active_frame
        cfg.room.max_frames = 150
    elif mode_choice == 2:
        cfg.room.start_frame = 0
        cfg.room.max_frames = 250
    elif mode_choice == 3:
        console.print("  [dim]Enter timestamp format as MM:SS (e.g. 01:00) or frame index (e.g. 1500):[/dim]")
        ts_input = Prompt.ask("Timestamp or Frame", default="01:00")
        if ":" in ts_input:
            parts = ts_input.split(":")
            seconds = int(parts[0]) * 60 + float(parts[1])
            cfg.room.start_frame = int(seconds * 25)
        else:
            cfg.room.start_frame = int(ts_input)
        cfg.room.max_frames = IntPrompt.ask("Frames to analyze from this point", default=150)
    elif mode_choice == 4:
        cfg.room.start_frame = 0
        cfg.room.max_frames = None

    # ── 3. Live Window View & Speed ────────────────────────────────────────────
    console.print("\n[bold yellow]Step 3: Visual Display & Playback Speed[/bold yellow]")
    console.print("  [dim]The Live HUD displays real-time demographic counts, hospital clock, and keyboard controls.[/dim]")
    live_view = Confirm.ask("Display live video detection window on screen?", default=True)
    cfg.artifacts.live_view = live_view

    if live_view:
        console.print("  [cyan][1][/cyan] [bold green]Accurate Continuous Inspection (Recommended)[/bold green] (Evaluates 100% of frames sequentially, highest recall)")
        console.print("  [cyan][2][/cyan] [yellow]Real-Time Camera Sync (1.0x)[/yellow] (Locks display playback 1:1 to real-world clock)")
        console.print("  [cyan][3][/cyan] [blue]Smooth Slow-Motion (0.5x)[/blue] (Detailed postural and movement inspection)")
        console.print("  [cyan][4][/cyan] [magenta]Fast-Forward (2.0x)[/magenta]")
        speed_choice = Prompt.ask("Choose playback mode", choices=["1", "2", "3", "4"], default="1")
        if speed_choice == "1":
            cfg.artifacts.playback_speed = 0.0
            cfg.artifacts.realtime_sync = False
        elif speed_choice == "2":
            cfg.artifacts.playback_speed = 1.0
            cfg.artifacts.realtime_sync = True
        elif speed_choice == "3":
            cfg.artifacts.playback_speed = 0.5
            cfg.artifacts.realtime_sync = False
        elif speed_choice == "4":
            cfg.artifacts.playback_speed = 2.0
            cfg.artifacts.realtime_sync = True
    else:
        cfg.artifacts.playback_speed = 0.0
        cfg.artifacts.realtime_sync = False

    # ── 4. Detector Model Engine ──────────────────────────────────────────────
    console.print("\n[bold yellow]Step 4: AI Model Engine (ONNX CPU Accelerated)[/bold yellow]")
    console.print("  [cyan][1][/cyan] [bold green]YOLO26s ONNX (Recommended)[/bold green] — 7.5–8.1 FPS, low CPU latency")
    console.print("  [cyan][2][/cyan] [blue]YOLO26m ONNX (High-Recall)[/blue]  — 2.0–2.7 FPS, maximum detection recall in crowds")
    console.print("  [cyan][3][/cyan] [dim]YOLO26s PyTorch (.pt)[/dim]       — PyTorch baseline")
    model_choice = Prompt.ask("Choose detector backend", choices=["1", "2", "3"], default="1")
    if model_choice == "1":
        cfg.models.detector_path = Path("computer_vision_models/yolo26/onnx/yolo26s.onnx")
    elif model_choice == "2":
        cfg.models.detector_path = Path("computer_vision_models/yolo26/onnx/yolo26m.onnx")
    else:
        cfg.models.detector_path = Path("computer_vision_models/yolo26/base_model/yolo26s.pt")

    # Use ONNX classifier if present
    onnx_cls = Path("computer_vision_models/yolo26/onnx/onnx_fine_tuned/pediatric-model.onnx")
    if onnx_cls.exists():
        cfg.models.classifier_path = onnx_cls

    # ── 5. Skeletal Pose Estimation Toggle ────────────────────────────────────
    console.print("\n[bold yellow]Step 5: Skeletal Pose & Anthropometric Ratio Analysis[/bold yellow]")
    console.print("  [dim]Extracts 17 COCO skeletal keypoints to calculate torso-to-leg ratios (>=0.80 child, <=0.70 adult).[/dim]")
    enable_pose = Confirm.ask("Enable 17-Point Skeletal Pose Estimation Plugin?", default=False)
    cfg.pose.enabled = enable_pose
    if enable_pose:
        console.print("  [cyan][1][/cyan] [bold green]YOLO26s Pose ONNX (Recommended)[/bold green] — Fast skeletal validation")
        console.print("  [cyan][2][/cyan] [blue]YOLO26m Pose ONNX (High-Precision)[/blue]  — Deeper multi-scale keypoint estimation")
        pose_choice = Prompt.ask("Choose pose model", choices=["1", "2"], default="1")
        if pose_choice == "1":
            cfg.pose.model_path = Path("computer_vision_models/yolo26/onnx/pose_models/yolo26s-pose.onnx")
        else:
            cfg.pose.model_path = Path("computer_vision_models/yolo26/onnx/pose_models/yolo26m-pose.onnx")

    # ── 6. Person Re-Identification (Re-ID) Plugin ───────────────────────────
    console.print("\n[bold yellow]Step 6: Person Re-Identification (Re-ID) & Multi-Camera Gallery[/bold yellow]")
    console.print("  [dim]Extracts dual-zone appearance signatures to recognize individuals across occlusions and multiple camera feeds.[/dim]")
    enable_reid = Confirm.ask("Enable Person Re-Identification (Re-ID) Plugin?", default=False)
    cfg.reid.enabled = enable_reid
    if enable_reid:
        console.print("  [cyan][1][/cyan] [bold green]Fast CPU Appearance Descriptor (Recommended)[/bold green] — Dual-zone HSV color histogram (<1ms)")
        console.print("  [cyan][2][/cyan] [blue]Deep Embedding Model (GPU/ONNX Ready)[/blue] — Pluggable 512-D neural feature extractor")
        reid_choice = Prompt.ask("Choose Re-ID backend", choices=["1", "2"], default="1")
        if reid_choice == "2":
            deep_path = Prompt.ask("Enter deep Re-ID ONNX/PT model path", default="computer_vision_models/reid/osnet_x0_25.onnx")
            cfg.reid.deep_model_path = deep_path
            cfg.reid.use_gpu = Confirm.ask("Enable GPU acceleration (CUDA) for Re-ID if available?", default=False)

    # ── 7. Demographic Counting ───────────────────────────────────────────────
    console.print("\n[bold yellow]Step 7: Demographic Counting[/bold yellow]")
    console.print("  [dim]The system always counts pediatric patients (children). You can also count accompanying adults.[/dim]")
    count_adults = Confirm.ask("Count Adults as well as Children?", default=True)
    cfg.counting.count_adults = count_adults

    # ── 8. Confirmation & Launch ───────────────────────────────────────────────
    start_time_str = f"{cfg.room.start_frame // 25 // 60:02d}:{(cfg.room.start_frame // 25) % 60:02d}"
    detector_name = cfg.models.detector_path.name
    sync_desc = "1:1 Real-Time Camera Sync" if cfg.artifacts.realtime_sync else f"{cfg.artifacts.playback_speed}x Speed"
    console.print()
    console.print(
        Panel(
            f"[bold]Input Feed:[/bold]             {chosen_label}\n"
            f"[bold]Start At:[/bold]               Frame {cfg.room.start_frame} (~{start_time_str} elapsed)\n"
            f"[bold]Frames Count:[/bold]           {cfg.room.max_frames or 'Entire Video'}\n"
            f"[bold]Stage 1 Detector:[/bold]       [bold cyan]{detector_name}[/bold cyan]\n"
            f"[bold]Demographic Classifier:[/bold] [bold cyan]{cfg.models.classifier_path.name if cfg.models.classifier_path else 'None'}[/bold cyan]\n"
            f"[bold]Skeletal Pose Plugin:[/bold]   {'[bold green]ACTIVE (17 COCO Keypoints, Live HUD)[/bold green]' if cfg.pose.enabled else '[dim]Disabled[/dim]'}\n"
            f"[bold]Person Re-ID Plugin:[/bold]    {'[bold green]ACTIVE (Multi-Camera Gallery)[/bold green]' if cfg.reid.enabled else '[dim]Disabled[/dim]'}\n"
            f"[bold]Playback Display:[/bold]       {'[bold green]Active (' + sync_desc + ')[/bold green]' if live_view else '[dim]Disabled (Headless)[/dim]'}\n"
            f"[bold]Demographics:[/bold]           Children {'+ Adults' if count_adults else 'Only'}\n"
            f"[bold]Output Folder:[/bold]          {cfg.artifacts.output_dir}",
            title="[bold green]Ready to Launch Pipeline[/bold green]",
            border_style="green",
        )
    )

    if Confirm.ask("Start analysis now?", default=True):
        console.print("\n[bold green]Launching pipeline...[/bold green]\n")
        run_pipeline(cfg)
        console.print()
        show_recent_timestamps()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        from pediatric_counter.app.cli import main as cli_main
        # Pass-through known top-level commands and flags; otherwise inject 'run'
        _passthrough = ("run", "count", "docs", "evaluate", "--help", "-h")
        if sys.argv[1] not in _passthrough:
            sys.argv.insert(1, "run")
        cli_main()
    else:
        main()
