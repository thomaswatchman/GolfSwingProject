"""
main.py
Golf Swing Tracker — with joint angle overlays and auto-pause at each phase.

Usage
-----
python main.py --source media/testswing.MP4
python main.py --source media/testswing.MP4 --save-video --save-csv
python main.py --source 0                        # live webcam (no pause)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.capture.video_capture import VideoCapture
from src.analysis.pose_estimator import PoseEstimator
from src.analysis.swing_metrics import compute_metrics
from src.analysis.swing_phase_detector import SwingPhaseDetector, SwingPhase, PHASE_COLORS
from src.visualization.visualizer import plot_swing_timeline


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Golf Swing Tracker")
    p.add_argument("--source", default="0")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--save-csv", action="store_true")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--pause-ms", type=int, default=800,
                   help="How long to freeze on each new phase (ms). Default 800.")
    return p


def _source(raw: str) -> int | str:
    try:
        return int(raw)
    except ValueError:
        return raw


# -----------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------

_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_WHITE     = (255, 255, 255)
_BLACK     = (0,   0,   0)
_YELLOW    = (0,   220, 220)
_DARKGREY  = (40,  40,  40)


def draw_joint_angles(frame: np.ndarray, landmarks: dict, metrics: dict) -> np.ndarray:
    """
    Draw each angle value directly next to its joint on the frame.
    Also draws arc indicators at elbows and knees.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    def px(name):
        x, y, _ = landmarks[name]
        return int(x * w), int(y * h)

    # ── Elbow angles ────────────────────────────────────────────────
    for side, elbow, angle_key in [
        ("L", "left_elbow",  "lead_arm_angle"),
        ("R", "right_elbow", "trail_arm_angle"),
    ]:
        if angle_key not in metrics:
            continue
        ex, ey = px(elbow)
        val = metrics[angle_key]
        label = f"{val:.0f}°"
        _draw_angle_label(out, ex, ey, label, offset=(-55, -10))

    # ── Knee angles ──────────────────────────────────────────────────
    for side, knee, angle_key in [
        ("L", "left_knee",  "lead_knee_flex"),
        ("R", "right_knee", "trail_knee_flex"),
    ]:
        if angle_key not in metrics:
            continue
        kx, ky = px(knee)
        val = metrics[angle_key]
        label = f"{val:.0f}°"
        _draw_angle_label(out, kx, ky, label, offset=(10, 0))

    # ── Spine tilt ───────────────────────────────────────────────────
    if "spine_tilt" in metrics:
        # Draw at mid-torso
        ls = px("left_shoulder")
        rs = px("right_shoulder")
        lh = px("left_hip")
        rh = px("right_hip")
        mid_shoulder = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
        mid_hip      = ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2)
        mid_torso    = ((mid_shoulder[0]+mid_hip[0])//2,
                        (mid_shoulder[1]+mid_hip[1])//2)
        _draw_angle_label(out, mid_torso[0], mid_torso[1],
                          f"Spine {metrics['spine_tilt']:.0f}°", offset=(12, 0))

    # ── Shoulder / hip rotation ──────────────────────────────────────
    if "shoulder_rotation" in metrics:
        ls = px("left_shoulder")
        rs = px("right_shoulder")
        mid = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2 - 18)
        _draw_angle_label(out, mid[0], mid[1],
                          f"Sh.rot {metrics['shoulder_rotation']:.0f}°",
                          offset=(-70, -18))

    if "hip_rotation" in metrics:
        lh = px("left_hip")
        rh = px("right_hip")
        mid = ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2)
        _draw_angle_label(out, mid[0], mid[1],
                          f"Hip {metrics['hip_rotation']:.0f}°",
                          offset=(-60, 18))

    return out


def _draw_angle_label(img, x, y, text, offset=(10, -10)):
    """Small pill-shaped label with dark background."""
    ox, oy = offset
    tx, ty = x + ox, y + oy
    (tw, th), bl = cv2.getTextSize(text, _FONT, 0.48, 1)
    pad = 3
    cv2.rectangle(img,
                  (tx - pad, ty - th - pad),
                  (tx + tw + pad, ty + pad),
                  _DARKGREY, -1)
    cv2.putText(img, text, (tx, ty), _FONT, 0.48, _YELLOW, 1, cv2.LINE_AA)


def draw_phase_banner(frame, phase, color):
    """Coloured phase label top-right."""
    h, w = frame.shape[:2]
    label = phase.value.upper()
    (tw, th), _ = cv2.getTextSize(label, _FONT, 0.85, 2)
    x = w - tw - 20
    cv2.rectangle(frame, (x - 10, 8), (w - 8, th + 22), _BLACK, -1)
    cv2.rectangle(frame, (x - 10, 8), (w - 8, th + 22), color, 2)
    cv2.putText(frame, label, (x, th + 14), _FONT, 0.85, color, 2, cv2.LINE_AA)


def draw_pause_indicator(frame, phase, color, pause_ms):
    """Flashes 'PHASE LOCKED' text when paused."""
    h, w = frame.shape[:2]
    msg = f"[ {phase.value.upper()} ]"
    (tw, th), _ = cv2.getTextSize(msg, _FONT, 1.1, 2)
    x = (w - tw) // 2
    y = h - 30
    cv2.rectangle(frame, (x - 12, y - th - 10), (x + tw + 12, y + 10), _BLACK, -1)
    cv2.rectangle(frame, (x - 12, y - th - 10), (x + tw + 12, y + 10), color, 2)
    cv2.putText(frame, msg, (x, y), _FONT, 1.1, color, 2, cv2.LINE_AA)


# -----------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    source   = _source(args.source)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    is_live  = isinstance(source, int)

    video_writer = None
    rows: list[dict] = []

    phase_detector = SwingPhaseDetector()
    prev_phase     = SwingPhase.UNKNOWN

    with VideoCapture(source) as cap, PoseEstimator() as estimator:
        fps = cap.fps if cap.fps > 0 else 30.0
        print(f"[INFO] Source : {source}")
        print(f"[INFO] FPS    : {fps:.1f}")
        print("[INFO] Press Q to quit  |  SPACE to unpause manually.")

        for frame_idx, frame in enumerate(cap.frames()):

            # ── Pose ──────────────────────────────────────────────────
            landmarks = estimator.process(frame)
            annotated = estimator.draw(frame)

            # ── Phase ─────────────────────────────────────────────────
            phase       = phase_detector.update(landmarks)
            phase_color = PHASE_COLORS[phase]
            phase_changed = (phase != prev_phase and
                             phase != SwingPhase.UNKNOWN)

            # ── Metrics ───────────────────────────────────────────────
            metrics: dict[str, float] = {}
            if landmarks:
                metrics = compute_metrics(landmarks)
                rows.append({"frame": frame_idx, "phase": phase.value, **metrics})

            # ── Draw joint angles directly on frame ───────────────────
            if landmarks and metrics:
                annotated = draw_joint_angles(annotated, landmarks, metrics)

            # ── Phase banner ──────────────────────────────────────────
            draw_phase_banner(annotated, phase, phase_color)

            # ── Video writer ──────────────────────────────────────────
            if args.save_video and video_writer is None:
                h, w = annotated.shape[:2]
                vpath = str(out_dir / "annotated_swing.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(vpath, fourcc, fps, (w, h))
                print(f"[INFO] Writing video → {vpath}")

            if video_writer:
                video_writer.write(annotated)

            # ── Display + pause logic ─────────────────────────────────
            if not args.no_display:
                cv2.imshow("Golf Swing Tracker  |  Q=quit  SPACE=unpause", annotated)

                # Auto-pause when a new phase begins (video files only)
                if phase_changed and not is_live:
                    pause_frame = annotated.copy()
                    draw_pause_indicator(pause_frame, phase, phase_color, args.pause_ms)
                    cv2.imshow("Golf Swing Tracker  |  Q=quit  SPACE=unpause", pause_frame)
                    # Wait pause_ms, but allow early unpause with SPACE or quit with Q
                    start = cv2.getTickCount()
                    while True:
                        elapsed_ms = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
                        if elapsed_ms >= args.pause_ms:
                            break
                        key = cv2.waitKey(30) & 0xFF
                        if key == ord("q"):
                            print("[INFO] Quit by user.")
                            if video_writer:
                                video_writer.release()
                            cv2.destroyAllWindows()
                            return
                        if key == ord(" "):
                            break

                else:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("[INFO] Quit by user.")
                        break

            prev_phase = phase

        # ── Teardown ──────────────────────────────────────────────────
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

    # ── Post-processing ───────────────────────────────────────────────
    if not rows:
        print("[WARN] No pose detections recorded.")
        return

    df = pd.DataFrame(rows)

    numeric_cols = [c for c in df.columns if c not in ("frame", "phase")]
    print("\n── Phase Summary (mean angles) ──────────────────────────────")
    print(df.groupby("phase")[numeric_cols].mean().round(1).to_string())
    print()

    if args.save_csv:
        csv_path = out_dir / "swing_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"[INFO] CSV → {csv_path}")

    fig = plot_swing_timeline(df, title="Swing Metric Timeline")
    plot_path = out_dir / "swing_timeline.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Plot → {plot_path}")


# -----------------------------------------------------------------------

if __name__ == "__main__":
    run(_build_parser().parse_args())