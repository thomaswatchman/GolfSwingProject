"""
main.py
CLI entry point — run the golf swing tracker on a video file or webcam.

Usage examples
--------------
python main.py --source media/GX016551.MP4
python main.py --source 0                        # live webcam
python main.py --source swing.mp4 --save-video --save-csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from src.capture.video_capture import VideoCapture
from src.analysis.pose_estimator import PoseEstimator
from src.analysis.swing_metrics import compute_metrics
from src.analysis.swing_phase_detector import SwingPhaseDetector, PHASE_COLORS
from src.visualization.visualizer import draw_metrics_overlay, plot_swing_timeline


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Golf Swing Tracker")
    p.add_argument("--source", default="0",
                   help="Camera index (0) or path to a video file")
    p.add_argument("--save-video", action="store_true",
                   help="Write annotated video to outputs/")
    p.add_argument("--save-csv", action="store_true",
                   help="Write per-frame metrics CSV to outputs/")
    p.add_argument("--output-dir", default="outputs",
                   help="Directory for saved files (default: outputs/)")
    p.add_argument("--no-display", action="store_true",
                   help="Suppress the live preview window")
    return p


def _source(raw: str) -> int | str:
    try:
        return int(raw)
    except ValueError:
        return raw


# -----------------------------------------------------------------------
# Overlay helper
# -----------------------------------------------------------------------

def _draw_phase_banner(frame, phase, color):
    """Draw a coloured phase label in the top-right corner."""
    h, w = frame.shape[:2]
    label = phase.value.upper()
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x = w - tw - 16
    cv2.rectangle(frame, (x - 8, 8), (w - 8, th + 20), (0, 0, 0), -1)
    cv2.rectangle(frame, (x - 8, 8), (w - 8, th + 20), color, 2)
    cv2.putText(frame, label, (x, th + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


# -----------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    source = _source(args.source)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    rows: list[dict] = []

    phase_detector = SwingPhaseDetector()

    with VideoCapture(source) as cap, PoseEstimator() as estimator:
        fps = cap.fps if cap.fps > 0 else 30.0
        print(f"[INFO] Source : {source}")
        print(f"[INFO] FPS    : {fps:.1f}")
        print(f"[INFO] Frames : {cap.frame_count if cap.frame_count > 0 else 'live'}")
        print("[INFO] Press Q to quit early.")

        for frame_idx, frame in enumerate(cap.frames()):

            # ── Pose ──────────────────────────────────────────────────
            landmarks = estimator.process(frame)
            annotated = estimator.draw(frame)

            # ── Phase detection ───────────────────────────────────────
            phase = phase_detector.update(landmarks)
            phase_color = PHASE_COLORS[phase]

            # ── Metrics ───────────────────────────────────────────────
            metrics: dict[str, float] = {}
            if landmarks:
                metrics = compute_metrics(landmarks)
                rows.append({
                    "frame": frame_idx,
                    "phase": phase.value,
                    **metrics,
                })

            # ── Overlay ───────────────────────────────────────────────
            annotated = draw_metrics_overlay(annotated, metrics, phase=phase.value)
            _draw_phase_banner(annotated, phase, phase_color)

            # ── Video writer ──────────────────────────────────────────
            if args.save_video and video_writer is None:
                h, w = annotated.shape[:2]
                vpath = str(out_dir / "annotated_swing.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(vpath, fourcc, fps, (w, h))
                print(f"[INFO] Writing video → {vpath}")

            if video_writer:
                video_writer.write(annotated)

            # ── Display ───────────────────────────────────────────────
            if not args.no_display:
                cv2.imshow("Golf Swing Tracker  |  Q = quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Quit by user.")
                    break

        # ── Teardown ──────────────────────────────────────────────────
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

    # ── Post-processing ───────────────────────────────────────────────
    if not rows:
        print("[WARN] No pose detections — nothing to save.")
        return

    df = pd.DataFrame(rows)

    # Per-phase summary printed to terminal
    numeric_cols = [c for c in df.columns if c not in ("frame", "phase")]
    print("\n── Phase Summary (mean values) ──────────────────────────────")
    summary = df.groupby("phase")[numeric_cols].mean().round(1)
    print(summary.to_string())
    print()

    if args.save_csv:
        csv_path = out_dir / "swing_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"[INFO] CSV saved → {csv_path}")

    # Timeline plot (colour-coded by phase)
    fig = plot_swing_timeline(df, title="Swing Metric Timeline")
    plot_path = out_dir / "swing_timeline.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Plot saved → {plot_path}")


# -----------------------------------------------------------------------

if __name__ == "__main__":
    run(_build_parser().parse_args())