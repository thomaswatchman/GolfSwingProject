"""
player.py
Web-based Golf Swing Player.

python player.py --source media/GX016551.MP4
Then open http://localhost:5100
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string

from src.analysis.pose_estimator import PoseEstimator
from src.analysis.swing_metrics import compute_metrics
from src.analysis.swing_phase_detector import SwingPhaseDetector, SwingPhase, PHASE_COLORS
from src.capture.video_capture import VideoCapture


# ═══════════════════════════════════════════════════════════════════════
# Frame rendering
# ═══════════════════════════════════════════════════════════════════════

_FONT   = cv2.FONT_HERSHEY_SIMPLEX
_YELLOW = (0, 220, 220)
_BLACK  = (0, 0, 0)


def _px(landmarks, name, w, h):
    x, y, _ = landmarks[name]
    return int(x * w), int(y * h)


def draw_angle_at(img, x, y, value: float, offset=(0, 0)):
    text = str(int(round(value)))
    ox, oy = offset
    tx, ty = x + ox, y + oy
    scale, thickness = 0.75, 2
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    pad = 5
    cv2.rectangle(img, (tx-pad, ty-th-pad), (tx+tw+pad, ty+pad), _BLACK, -1)
    cv2.putText(img, text, (tx, ty), _FONT, scale, _YELLOW, thickness, cv2.LINE_AA)


def draw_angles(frame, landmarks, metrics):
    out = frame.copy()
    h, w = out.shape[:2]
    def px(name): return _px(landmarks, name, w, h)

    if "lead_arm_angle"    in metrics:
        ex, ey = px("left_elbow")
        draw_angle_at(out, ex, ey, metrics["lead_arm_angle"],    offset=(-50, -14))
    if "trail_arm_angle"   in metrics:
        ex, ey = px("right_elbow")
        draw_angle_at(out, ex, ey, metrics["trail_arm_angle"],   offset=(12, -14))
    if "lead_knee_flex"    in metrics:
        kx, ky = px("left_knee")
        draw_angle_at(out, kx, ky, metrics["lead_knee_flex"],    offset=(-50, -14))
    if "trail_knee_flex"   in metrics:
        kx, ky = px("right_knee")
        draw_angle_at(out, kx, ky, metrics["trail_knee_flex"],   offset=(12, -14))
    if "spine_tilt"        in metrics:
        ls, rs = px("left_shoulder"), px("right_shoulder")
        lh, rh = px("left_hip"),      px("right_hip")
        mx = (ls[0]+rs[0]+lh[0]+rh[0])//4
        my = (ls[1]+rs[1]+lh[1]+rh[1])//4
        draw_angle_at(out, mx, my, metrics["spine_tilt"],        offset=(14, 0))
    if "shoulder_rotation" in metrics:
        ls, rs = px("left_shoulder"), px("right_shoulder")
        mx = (ls[0]+rs[0])//2
        my = min(ls[1], rs[1]) - 24
        draw_angle_at(out, mx, my, metrics["shoulder_rotation"], offset=(-25, 0))
    if "hip_rotation"      in metrics:
        lh, rh = px("left_hip"), px("right_hip")
        mx = (lh[0]+rh[0])//2
        my = max(lh[1], rh[1]) + 28
        draw_angle_at(out, mx, my, metrics["hip_rotation"],      offset=(-25, 0))
    return out


def draw_phase_banner(frame, phase, color):
    h, w = frame.shape[:2]
    label = phase.value.upper()
    (tw, th), _ = cv2.getTextSize(label, _FONT, 0.85, 2)
    x = w - tw - 20
    cv2.rectangle(frame, (x-10, 8), (w-8, th+22), _BLACK, -1)
    cv2.rectangle(frame, (x-10, 8), (w-8, th+22), color, 2)
    cv2.putText(frame, label, (x, th+14), _FONT, 0.85, color, 2, cv2.LINE_AA)


def encode_frame(frame) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode()


# ═══════════════════════════════════════════════════════════════════════
# Video processing
# ═══════════════════════════════════════════════════════════════════════

def process_video(source) -> tuple[list[dict], float]:
    frames_data   = []
    phase_detector = SwingPhaseDetector()

    with VideoCapture(source) as cap, PoseEstimator() as estimator:
        fps   = cap.fps if cap.fps > 0 else 30.0
        total = cap.frame_count
        print(f"[INFO] Processing {total if total > 0 else '?'} frames at {fps:.1f} fps ...")

        for i, frame in enumerate(cap.frames()):
            if i % 60 == 0:
                print(f"  frame {i}" + (f"/{total}" if total > 0 else ""), end="\r", flush=True)

            landmarks = estimator.process(frame)
            annotated = estimator.draw(frame)
            phase     = phase_detector.update(landmarks)
            color     = PHASE_COLORS[phase]

            metrics: dict[str, float] = {}
            if landmarks:
                metrics   = compute_metrics(landmarks)
                annotated = draw_angles(annotated, landmarks, metrics)

            draw_phase_banner(annotated, phase, color)

            frames_data.append({
                "img_b64": encode_frame(annotated),
                "metrics": {k: round(v, 1) for k, v in metrics.items()},
                "phase":   phase.value,
            })

    print(f"\n[INFO] Done - {len(frames_data)} frames processed.")
    return frames_data, fps


# ═══════════════════════════════════════════════════════════════════════
# HTML
# ═══════════════════════════════════════════════════════════════════════

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SwingTrack</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #080a0e;
    --surface:  #10131a;
    --surface2: #181c26;
    --border:   #1f2535;
    --accent:   #00d4ff;
    --accent2:  #ff5533;
    --text:     #dde2ee;
    --muted:    #4a5268;
    --setup:    #8899aa;
    --back:     #22dd66;
    --trans:    #ffcc00;
    --down:     #ff3344;
    --follow:   #ff7722;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Mono', monospace;
    min-height: 100vh;
    display: flex; flex-direction: column; align-items: center;
    padding: 20px 16px 60px;
  }

  header {
    width: 100%; max-width: 1200px;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 20px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 14px;
  }
  .logo {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.2rem; font-weight: 700;
    letter-spacing: 0.14em; color: var(--accent);
  }
  .logo span { color: var(--accent2); }
  .tagline { font-size: 0.65rem; color: var(--muted); letter-spacing: 0.15em; text-transform: uppercase; }
  .kbd-hints { font-size: 0.62rem; color: var(--muted); text-align: right; line-height: 1.9; }

  .layout {
    width: 100%; max-width: 1200px;
    display: grid;
    grid-template-columns: 1fr 300px;
    gap: 16px;
  }

  .video-col { display: flex; flex-direction: column; gap: 12px; }

  #video-wrap {
    background: #000;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden; line-height: 0;
  }
  #frame-img { width: 100%; display: block; }

  .controls {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 16px;
    display: flex; flex-direction: column; gap: 10px;
  }
  .ctrl-row { display: flex; align-items: center; gap: 10px; }

  #btn-play {
    width: 42px; height: 42px; flex-shrink: 0;
    border: 2px solid var(--accent); border-radius: 50%;
    background: transparent; color: var(--accent);
    font-size: 1rem; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.15s;
  }
  #btn-play:hover { background: var(--accent); color: var(--bg); }

  #scrubber {
    flex: 1; -webkit-appearance: none;
    height: 3px; border-radius: 2px;
    background: var(--border); outline: none; cursor: pointer;
  }
  #scrubber::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 13px; height: 13px; border-radius: 50%;
    background: var(--accent); cursor: pointer;
  }
  #frame-counter { font-size: 0.65rem; color: var(--muted); min-width: 76px; text-align: right; }

  .speed-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .speed-label { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }
  .speed-btn {
    padding: 3px 9px;
    border: 1px solid var(--border); border-radius: 3px;
    background: transparent; color: var(--muted);
    font-family: 'DM Mono', monospace; font-size: 0.7rem;
    cursor: pointer; transition: all 0.12s;
  }
  .speed-btn:hover  { border-color: var(--accent); color: var(--accent); }
  .speed-btn.active { background: var(--accent); border-color: var(--accent); color: var(--bg); }

  .custom-speed-wrap {
    display: flex; align-items: center; gap: 6px;
    margin-left: auto;
  }
  .custom-speed-wrap label { font-size: 0.65rem; color: var(--muted); white-space: nowrap; }
  #custom-speed-input {
    width: 60px; padding: 3px 7px;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 3px; color: var(--text);
    font-family: 'DM Mono', monospace; font-size: 0.7rem; outline: none;
  }
  #custom-speed-input:focus { border-color: var(--accent); }
  #btn-set-speed {
    padding: 3px 9px;
    border: 1px solid var(--accent2); border-radius: 3px;
    background: transparent; color: var(--accent2);
    font-family: 'DM Mono', monospace; font-size: 0.7rem;
    cursor: pointer; transition: all 0.12s;
  }
  #btn-set-speed:hover { background: var(--accent2); color: var(--bg); }

  .phase-jumps {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 16px;
    display: flex; flex-direction: column; gap: 10px;
  }
  .section-title {
    font-size: 0.6rem; text-transform: uppercase;
    letter-spacing: 0.14em; color: var(--muted);
  }
  .phase-btn-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
  }
  .phase-jump-btn {
    padding: 9px 4px;
    border-radius: 4px; border: 1px solid;
    background: transparent;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.82rem; font-weight: 600;
    letter-spacing: 0.05em;
    cursor: pointer; text-align: center;
    transition: all 0.15s; line-height: 1.2;
  }
  .phase-jump-btn:disabled { opacity: 0.2; cursor: not-allowed; }
  .phase-jump-btn.pj-setup  { border-color: var(--setup);  color: var(--setup); }
  .phase-jump-btn.pj-back   { border-color: var(--back);   color: var(--back); }
  .phase-jump-btn.pj-trans  { border-color: var(--trans);  color: var(--trans); }
  .phase-jump-btn.pj-down   { border-color: var(--down);   color: var(--down); }
  .phase-jump-btn.pj-follow { border-color: var(--follow); color: var(--follow); }
  .phase-jump-btn.pj-setup:hover:not(:disabled)  { background: var(--setup);  color: var(--bg); }
  .phase-jump-btn.pj-back:hover:not(:disabled)   { background: var(--back);   color: var(--bg); }
  .phase-jump-btn.pj-trans:hover:not(:disabled)  { background: var(--trans);  color: var(--bg); }
  .phase-jump-btn.pj-down:hover:not(:disabled)   { background: var(--down);   color: var(--bg); }
  .phase-jump-btn.pj-follow:hover:not(:disabled) { background: var(--follow); color: var(--bg); }
  .phase-jump-btn.is-active-phase { box-shadow: 0 0 0 2px currentColor; }

  .sidebar { display: flex; flex-direction: column; gap: 12px; }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px; padding: 16px;
  }
  .card .section-title { margin-bottom: 12px; }

  #phase-badge {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.9rem; font-weight: 700;
    letter-spacing: 0.1em; transition: color 0.2s;
  }
  .pc-setup   { color: var(--setup); }
  .pc-back    { color: var(--back); }
  .pc-trans   { color: var(--trans); }
  .pc-down    { color: var(--down); }
  .pc-follow  { color: var(--follow); }
  .pc-unknown { color: var(--muted); }

  .metric-row {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 5px 0; border-bottom: 1px solid var(--border); font-size: 0.7rem;
  }
  .metric-row:last-child { border-bottom: none; }
  .metric-name  { color: var(--muted); }
  .metric-value { font-size: 0.95rem; font-weight: 500; color: var(--text); }

  .phase-table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 0.62rem; }
  th, td {
    padding: 5px 6px; text-align: right;
    border-bottom: 1px solid var(--border); white-space: nowrap;
  }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--muted); font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.58rem; }
  td { color: var(--text); }
  tr:last-child td { border-bottom: none; }
  .ph-setup  td:not(:first-child) { color: var(--setup); }
  .ph-back   td:not(:first-child) { color: var(--back); }
  .ph-trans  td:not(:first-child) { color: var(--trans); }
  .ph-down   td:not(:first-child) { color: var(--down); }
  .ph-follow td:not(:first-child) { color: var(--follow); }

  #loading {
    position: fixed; inset: 0;
    background: var(--bg);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 14px; z-index: 999;
  }
  #loading.hidden { display: none; }
  .spinner {
    width: 34px; height: 34px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.75s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  #load-msg { font-size: 0.85rem; color: var(--muted); }
</style>
</head>
<body>

<div id="loading">
  <div class="spinner"></div>
  <div id="load-msg">Loading swing data...</div>
</div>

<header>
  <div>
    <div class="logo">SWING<span>TRACK</span></div>
    <div class="tagline">Golf Swing Analyser</div>
  </div>
  <div class="kbd-hints">
    SPACE &nbsp;play / pause<br>
    &larr; &rarr; &nbsp;step frame
  </div>
</header>

<div class="layout">

  <div class="video-col">

    <div id="video-wrap">
      <img id="frame-img" src="" alt="swing frame">
    </div>

    <div class="controls">
      <div class="ctrl-row">
        <button id="btn-play" title="Play / Pause">&#x25B6;</button>
        <input id="scrubber" type="range" min="0" value="0" step="1">
        <span id="frame-counter">0 / 0</span>
      </div>
      <div class="ctrl-row speed-row">
        <span class="speed-label">Speed</span>
        <button class="speed-btn" data-speed="0.25">0.25x</button>
        <button class="speed-btn" data-speed="0.5">0.5x</button>
        <button class="speed-btn active" data-speed="1">1x</button>
        <button class="speed-btn" data-speed="1.5">1.5x</button>
        <button class="speed-btn" data-speed="2">2x</button>
        <div class="custom-speed-wrap">
          <label for="custom-speed-input">Custom</label>
          <input id="custom-speed-input" type="number" min="0.1" max="10" step="0.1" placeholder="e.g. 3">
          <button id="btn-set-speed">Set</button>
        </div>
      </div>
    </div>

    <div class="phase-jumps">
      <div class="section-title">Jump to Phase</div>
      <div class="phase-btn-grid">
        <button class="phase-jump-btn pj-setup"  data-phase="Setup"          disabled>SETUP</button>
        <button class="phase-jump-btn pj-back"   data-phase="Backswing"      disabled>BACK&#x00AD;SWING</button>
        <button class="phase-jump-btn pj-trans"  data-phase="Transition"     disabled>TRANS&#x00AD;ITION</button>
        <button class="phase-jump-btn pj-down"   data-phase="Downswing"      disabled>DOWN&#x00AD;SWING</button>
        <button class="phase-jump-btn pj-follow" data-phase="Follow Through" disabled>FOLLOW THRU</button>
      </div>
    </div>

  </div>

  <div class="sidebar">

    <div class="card">
      <div class="section-title">Current Phase</div>
      <div id="phase-badge">—</div>
    </div>

    <div class="card">
      <div class="section-title">Joint Angles — Current Frame</div>
      <div id="metrics-list"></div>
    </div>

    <div class="card">
      <div class="section-title">Avg Angles by Phase</div>
      <div class="phase-table-wrap">
        <table id="phase-table">
          <thead>
            <tr>
              <th>Phase</th>
              <th>L.Arm</th>
              <th>R.Arm</th>
              <th>Sh.Rot</th>
              <th>Hip</th>
              <th>Spine</th>
              <th>L.Kn</th>
              <th>R.Kn</th>
            </tr>
          </thead>
          <tbody id="phase-table-body">
            <tr><td colspan="8" style="color:var(--muted);text-align:center">Loading...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

  </div>
</div>

<script>
const METRIC_LABELS = {
  lead_arm_angle:    'Lead Arm',
  trail_arm_angle:   'Trail Arm',
  shoulder_rotation: 'Sh. Rotation',
  hip_rotation:      'Hip Rotation',
  spine_tilt:        'Spine Tilt',
  lead_knee_flex:    'Lead Knee',
  trail_knee_flex:   'Trail Knee',
};
const METRIC_KEYS = Object.keys(METRIC_LABELS);

const PHASE_COLORS = {
  'Setup':          'pc-setup',
  'Backswing':      'pc-back',
  'Transition':     'pc-trans',
  'Downswing':      'pc-down',
  'Follow Through': 'pc-follow',
  'Unknown':        'pc-unknown',
};
const PHASE_ROW_CLASS = {
  'Setup':          'ph-setup',
  'Backswing':      'ph-back',
  'Transition':     'ph-trans',
  'Downswing':      'ph-down',
  'Follow Through': 'ph-follow',
};
const PHASE_ORDER = ['Setup','Backswing','Transition','Downswing','Follow Through'];

let FRAMES = [], FPS = 30;
let phaseIndex = {};

const imgEl          = document.getElementById('frame-img');
const scrubber       = document.getElementById('scrubber');
const counter        = document.getElementById('frame-counter');
const btnPlay        = document.getElementById('btn-play');
const phaseBadge     = document.getElementById('phase-badge');
const metricsList    = document.getElementById('metrics-list');
const loadingEl      = document.getElementById('loading');
const loadMsg        = document.getElementById('load-msg');
const speedBtns      = document.querySelectorAll('.speed-btn');
const phaseJumpBtns  = document.querySelectorAll('.phase-jump-btn');
const phaseTableBody = document.getElementById('phase-table-body');
const customInput    = document.getElementById('custom-speed-input');
const btnSetSpeed    = document.getElementById('btn-set-speed');

let currentFrame = 0, playing = false, speed = 1.0;
let lastTime = null, accumMs = 0;

function showFrame(idx) {
  currentFrame = Math.max(0, Math.min(idx, FRAMES.length - 1));
  const f = FRAMES[currentFrame];
  imgEl.src = 'data:image/jpeg;base64,' + f.img_b64;
  scrubber.value = currentFrame;
  counter.textContent = currentFrame + ' / ' + (FRAMES.length - 1);

  const phase = f.phase || 'Unknown';
  phaseBadge.textContent = phase.toUpperCase();
  phaseBadge.className = PHASE_COLORS[phase] || 'pc-unknown';

  phaseJumpBtns.forEach(btn => {
    btn.classList.toggle('is-active-phase', btn.dataset.phase === phase);
  });

  const m = f.metrics || {};
  metricsList.innerHTML = METRIC_KEYS.map(key => {
    const val = m[key] !== undefined ? m[key].toFixed(1) + '\u00b0' : '\u2014';
    return '<div class="metric-row"><span class="metric-name">' +
      METRIC_LABELS[key] + '</span><span class="metric-value">' + val + '</span></div>';
  }).join('');
}

function tick(ts) {
  if (!playing) return;
  if (lastTime !== null) {
    accumMs += (ts - lastTime) * speed;
    const mpf = 1000 / FPS;
    while (accumMs >= mpf) {
      accumMs -= mpf;
      if (currentFrame >= FRAMES.length - 1) { pause(); showFrame(0); return; }
      showFrame(currentFrame + 1);
    }
  }
  lastTime = ts;
  requestAnimationFrame(tick);
}

function play() {
  if (!FRAMES.length) return;
  playing = true; lastTime = null; accumMs = 0;
  btnPlay.innerHTML = '&#x23F8;';
  requestAnimationFrame(tick);
}

function pause() {
  playing = false;
  btnPlay.innerHTML = '&#x25B6;';
}

function setSpeed(val) {
  const s = parseFloat(val);
  if (isNaN(s) || s <= 0) return;
  speed = s;
  speedBtns.forEach(b => {
    b.classList.toggle('active', parseFloat(b.dataset.speed) === s);
  });
}

btnPlay.addEventListener('click', () => playing ? pause() : play());
scrubber.addEventListener('input', () => { pause(); showFrame(parseInt(scrubber.value)); });

speedBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    setSpeed(btn.dataset.speed);
    customInput.value = '';
  });
});

btnSetSpeed.addEventListener('click', () => {
  const v = parseFloat(customInput.value);
  if (!isNaN(v) && v > 0) {
    speed = v;
    speedBtns.forEach(b => b.classList.remove('active'));
  }
});
customInput.addEventListener('keydown', e => { if (e.key === 'Enter') btnSetSpeed.click(); });

phaseJumpBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const idx = phaseIndex[btn.dataset.phase];
    if (idx !== undefined) { pause(); showFrame(idx); }
  });
});

document.addEventListener('keydown', e => {
  if (e.target === customInput) return;
  if (e.code === 'Space')      { e.preventDefault(); playing ? pause() : play(); }
  if (e.code === 'ArrowRight') { pause(); showFrame(currentFrame + 1); }
  if (e.code === 'ArrowLeft')  { pause(); showFrame(currentFrame - 1); }
});

function buildPhaseTable(frames) {
  const sums = {}, counts = {};
  PHASE_ORDER.forEach(p => { sums[p] = {}; counts[p] = 0; });
  frames.forEach(f => {
    const p = f.phase;
    if (!sums[p]) return;
    counts[p]++;
    METRIC_KEYS.forEach(k => {
      if (f.metrics[k] !== undefined)
        sums[p][k] = (sums[p][k] || 0) + f.metrics[k];
    });
  });
  phaseTableBody.innerHTML = PHASE_ORDER.map(phase => {
    const n = counts[phase];
    const cls = PHASE_ROW_CLASS[phase] || '';
    const cells = METRIC_KEYS.map(k => {
      if (!n || sums[phase][k] === undefined) return '<td>\u2014</td>';
      return '<td>' + (sums[phase][k] / n).toFixed(1) + '</td>';
    }).join('');
    return '<tr class="' + cls + '"><td>' + phase + '</td>' + cells + '</tr>';
  }).join('');
}

function buildPhaseIndex(frames) {
  frames.forEach((f, i) => {
    if (f.phase && phaseIndex[f.phase] === undefined)
      phaseIndex[f.phase] = i;
  });
  phaseJumpBtns.forEach(btn => {
    if (phaseIndex[btn.dataset.phase] !== undefined)
      btn.disabled = false;
  });
}

fetch('/frames')
  .then(r => r.json())
  .then(data => {
    FRAMES = data.frames;
    FPS    = data.fps;
    scrubber.max = FRAMES.length - 1;
    buildPhaseIndex(FRAMES);
    buildPhaseTable(FRAMES);
    loadingEl.classList.add('hidden');
    showFrame(0);
  })
  .catch(err => { loadMsg.textContent = 'Error: ' + err; });
</script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════════════════
# Flask
# ═══════════════════════════════════════════════════════════════════════

app = Flask(__name__)
_frames_data: list[dict] = []
_fps: float = 30.0


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/frames")
def frames():
    return jsonify({"frames": _frames_data, "fps": _fps})


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

def main():
    global _frames_data, _fps

    parser = argparse.ArgumentParser(description="Golf Swing Web Player")
    parser.add_argument("--source", required=True)
    parser.add_argument("--port", type=int, default=5100)
    args = parser.parse_args()

    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    _frames_data, _fps = process_video(source)
    print(f"\n[INFO] Open your browser -> http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()