
"""
visualizer.py
Draws metrics overlays on frames and plots time-series swing data.
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GREEN = (50, 205, 50)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)


def draw_metrics_overlay(
    frame: np.ndarray,
    metrics: dict[str, float],
    phase: str | None = None,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (260, h), _BLACK, -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
    if phase:
        cv2.putText(out, phase.upper(), (8, 30), _FONT, 0.7, _GREEN, 2, cv2.LINE_AA)
    y = 60
    for name, value in metrics.items():
        label = name.replace("_", " ").title()
        cv2.putText(out, f"{label}: {value:.1f}°", (8, y), _FONT, 0.48, _WHITE, 1, cv2.LINE_AA)
        y += 24
    return out


def plot_swing_timeline(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    title: str = "Golf Swing – Metric Timeline",
    save_path: str | None = None,
) -> plt.Figure:
    if metrics is None:
        metrics = [c for c in df.columns if c != "frame"]
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    colors = plt.cm.tab10.colors
    for ax, metric, color in zip(axes, metrics, colors):
        ax.plot(df["frame"], df[metric], color=color, linewidth=1.8, label=metric)
        ax.set_ylabel(f"{metric}\n(°)", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Frame")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
