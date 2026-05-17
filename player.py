"""
player.py  —  SwingTrack web player with swing library.

python player.py --source media/GX016551.MP4
Then open http://localhost:5100

Tabs:
  Player  — analyse the current video as before
  Library — browse saved swings, compare two side-by-side
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request

from src.analysis.pose_estimator import PoseEstimator
from src.analysis.swing_metrics import compute_metrics
from src.analysis.swing_phase_detector import SwingPhaseDetector, SwingPhase, PHASE_COLORS
from src.capture.video_capture import VideoCapture
from library import SwingLibrary


# ═══════════════════════════════════════════════════════════════════════
# Frame rendering helpers
# ═══════════════════════════════════════════════════════════════════════

_FONT   = cv2.FONT_HERSHEY_SIMPLEX
_YELLOW = (0, 220, 220)
_BLACK  = (0, 0, 0)


def _px(landmarks, name, w, h):
    x, y, _ = landmarks[name]
    return int(x * w), int(y * h)


def draw_angle_at(img, x, y, value, offset=(0, 0)):
    text = str(int(round(value)))
    tx, ty = x + offset[0], y + offset[1]
    scale, thick = 0.75, 2
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thick)
    pad = 5
    cv2.rectangle(img, (tx-pad, ty-th-pad), (tx+tw+pad, ty+pad), _BLACK, -1)
    cv2.putText(img, text, (tx, ty), _FONT, scale, _YELLOW, thick, cv2.LINE_AA)


def draw_angles(frame, landmarks, metrics):
    out = frame.copy()
    h, w = out.shape[:2]
    def px(n): return _px(landmarks, n, w, h)
    if "lead_arm_angle"    in metrics: draw_angle_at(out, *px("left_elbow"),   metrics["lead_arm_angle"],    (-50,-14))
    if "trail_arm_angle"   in metrics: draw_angle_at(out, *px("right_elbow"),  metrics["trail_arm_angle"],   ( 12,-14))
    if "lead_knee_flex"    in metrics: draw_angle_at(out, *px("left_knee"),    metrics["lead_knee_flex"],    (-50,-14))
    if "trail_knee_flex"   in metrics: draw_angle_at(out, *px("right_knee"),   metrics["trail_knee_flex"],   ( 12,-14))
    if "spine_tilt"        in metrics:
        ls,rs,lh,rh = px("left_shoulder"),px("right_shoulder"),px("left_hip"),px("right_hip")
        draw_angle_at(out,(ls[0]+rs[0]+lh[0]+rh[0])//4,(ls[1]+rs[1]+lh[1]+rh[1])//4,metrics["spine_tilt"],(14,0))
    if "shoulder_rotation" in metrics:
        ls,rs = px("left_shoulder"),px("right_shoulder")
        draw_angle_at(out,(ls[0]+rs[0])//2,min(ls[1],rs[1])-24,metrics["shoulder_rotation"],(-25,0))
    if "hip_rotation"      in metrics:
        lh,rh = px("left_hip"),px("right_hip")
        draw_angle_at(out,(lh[0]+rh[0])//2,max(lh[1],rh[1])+28,metrics["hip_rotation"],(-25,0))
    return out


def draw_phase_banner(frame, phase, color):
    h, w = frame.shape[:2]
    label = phase.value.upper()
    (tw, th), _ = cv2.getTextSize(label, _FONT, 0.85, 2)
    x = w - tw - 20
    cv2.rectangle(frame, (x-10,8), (w-8,th+22), _BLACK, -1)
    cv2.rectangle(frame, (x-10,8), (w-8,th+22), color, 2)
    cv2.putText(frame, label, (x, th+14), _FONT, 0.85, color, 2, cv2.LINE_AA)


def encode_frame(frame) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode()


# ═══════════════════════════════════════════════════════════════════════
# Video processing
# ═══════════════════════════════════════════════════════════════════════

def process_video(source) -> tuple[list[dict], float]:
    frames_data    = []
    phase_detector = SwingPhaseDetector()

    with VideoCapture(source) as cap, PoseEstimator() as estimator:
        fps   = cap.fps if cap.fps > 0 else 30.0
        total = cap.frame_count
        print(f"[INFO] Processing {total if total>0 else '?'} frames at {fps:.1f} fps ...")

        for i, frame in enumerate(cap.frames()):
            if i % 60 == 0:
                print(f"  frame {i}" + (f"/{total}" if total>0 else ""), end="\r", flush=True)

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
                "metrics": {k: round(v,1) for k,v in metrics.items()},
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
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }
:root {
  --bg:#080a0e; --surface:#10131a; --surface2:#181c26;
  --border:#1f2535; --accent:#00d4ff; --accent2:#ff5533;
  --text:#dde2ee; --muted:#4a5268;
  --back:#22dd66; --down:#ff3344; --contact:#00ffff; --follow:#ff7722;
}
body { background:var(--bg); color:var(--text); font-family:'DM Mono',monospace;
  min-height:100vh; display:flex; flex-direction:column; align-items:center; padding:20px 16px 60px; }

header { width:100%; max-width:1300px; display:flex; align-items:center; justify-content:space-between;
  margin-bottom:0; border-bottom:1px solid var(--border); padding-bottom:14px; }
.logo { font-family:'Barlow Condensed',sans-serif; font-size:2.2rem; font-weight:700; letter-spacing:.14em; color:var(--accent); }
.logo span { color:var(--accent2); }
.tagline { font-size:.65rem; color:var(--muted); letter-spacing:.15em; text-transform:uppercase; }
.kbd-hints { font-size:.62rem; color:var(--muted); text-align:right; line-height:1.9; }

.tabs { width:100%; max-width:1300px; display:flex; margin-bottom:20px; border-bottom:1px solid var(--border); }
.tab-btn { padding:12px 28px; background:transparent; border:none; border-bottom:2px solid transparent;
  color:var(--muted); font-family:'DM Mono',monospace; font-size:.78rem; letter-spacing:.08em;
  cursor:pointer; text-transform:uppercase; transition:all .15s; margin-bottom:-1px; }
.tab-btn:hover { color:var(--text); }
.tab-btn.active { color:var(--accent); border-bottom-color:var(--accent); }
.tab-panel { display:none; width:100%; max-width:1300px; }
.tab-panel.active { display:block; }

.card { background:var(--surface); border:1px solid var(--border); border-radius:6px; padding:16px; }
.section-title { font-size:.6rem; text-transform:uppercase; letter-spacing:.14em; color:var(--muted); margin-bottom:12px; }

/* ── Player ── */
.player-layout { display:grid; grid-template-columns:1fr 300px; gap:16px; }
.video-col { display:flex; flex-direction:column; gap:12px; }
#video-wrap { background:#000; border:1px solid var(--border); border-radius:6px; overflow:hidden; line-height:0; }
#frame-img { width:100%; display:block; }

.controls { background:var(--surface); border:1px solid var(--border); border-radius:6px; padding:14px 16px; display:flex; flex-direction:column; gap:10px; }
.ctrl-row { display:flex; align-items:center; gap:10px; }
#btn-play { width:42px; height:42px; flex-shrink:0; border:2px solid var(--accent); border-radius:50%;
  background:transparent; color:var(--accent); font-size:1rem; cursor:pointer;
  display:flex; align-items:center; justify-content:center; transition:all .15s; }
#btn-play:hover { background:var(--accent); color:var(--bg); }
#scrubber { flex:1; -webkit-appearance:none; height:3px; border-radius:2px; background:var(--border); outline:none; cursor:pointer; }
#scrubber::-webkit-slider-thumb { -webkit-appearance:none; width:13px; height:13px; border-radius:50%; background:var(--accent); cursor:pointer; }
#frame-counter { font-size:.65rem; color:var(--muted); min-width:76px; text-align:right; }
.speed-row { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.speed-label { font-size:.65rem; color:var(--muted); text-transform:uppercase; letter-spacing:.1em; }
.speed-btn { padding:3px 9px; border:1px solid var(--border); border-radius:3px; background:transparent; color:var(--muted);
  font-family:'DM Mono',monospace; font-size:.7rem; cursor:pointer; transition:all .12s; }
.speed-btn:hover { border-color:var(--accent); color:var(--accent); }
.speed-btn.active { background:var(--accent); border-color:var(--accent); color:var(--bg); }
.custom-speed-wrap { display:flex; align-items:center; gap:6px; margin-left:auto; }
.custom-speed-wrap label { font-size:.65rem; color:var(--muted); white-space:nowrap; }
#custom-speed-input { width:60px; padding:3px 7px; background:var(--surface2); border:1px solid var(--border);
  border-radius:3px; color:var(--text); font-family:'DM Mono',monospace; font-size:.7rem; outline:none; }
#custom-speed-input:focus { border-color:var(--accent); }
#btn-set-speed { padding:3px 9px; border:1px solid var(--accent2); border-radius:3px; background:transparent; color:var(--accent2);
  font-family:'DM Mono',monospace; font-size:.7rem; cursor:pointer; transition:all .12s; }
#btn-set-speed:hover { background:var(--accent2); color:var(--bg); }

.phase-jumps { background:var(--surface); border:1px solid var(--border); border-radius:6px; padding:14px 16px; display:flex; flex-direction:column; gap:10px; }
.phase-btn-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; }
.phase-jump-btn { padding:9px 4px; border-radius:4px; border:1px solid; background:transparent;
  font-family:'Barlow Condensed',sans-serif; font-size:.82rem; font-weight:600;
  cursor:pointer; text-align:center; transition:all .15s; line-height:1.2; }
.phase-jump-btn:disabled { opacity:.2; cursor:not-allowed; }
.phase-jump-btn.pj-back    { border-color:var(--back);    color:var(--back); }
.phase-jump-btn.pj-down    { border-color:var(--down);    color:var(--down); }
.phase-jump-btn.pj-contact { border-color:var(--contact); color:var(--contact); }
.phase-jump-btn.pj-follow  { border-color:var(--follow);  color:var(--follow); }
.phase-jump-btn.pj-back:hover:not(:disabled)    { background:var(--back);    color:var(--bg); }
.phase-jump-btn.pj-down:hover:not(:disabled)    { background:var(--down);    color:var(--bg); }
.phase-jump-btn.pj-contact:hover:not(:disabled) { background:var(--contact); color:var(--bg); }
.phase-jump-btn.pj-follow:hover:not(:disabled)  { background:var(--follow);  color:var(--bg); }
.phase-jump-btn.is-active-phase { box-shadow:0 0 0 2px currentColor; }

.sidebar { display:flex; flex-direction:column; gap:12px; }
#phase-badge { font-family:'Barlow Condensed',sans-serif; font-size:1.9rem; font-weight:700; letter-spacing:.1em; transition:color .2s; }
.pc-back{color:var(--back);} .pc-down{color:var(--down);} .pc-contact{color:var(--contact);} .pc-follow{color:var(--follow);} .pc-unknown{color:var(--muted);}

.metric-row { display:flex; justify-content:space-between; align-items:baseline;
  padding:5px 0; border-bottom:1px solid var(--border); font-size:.7rem; }
.metric-row:last-child { border-bottom:none; }
.metric-name { color:var(--muted); }
.metric-value { font-size:.95rem; font-weight:500; color:var(--text); }

.phase-table-wrap { overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-size:.62rem; }
th, td { padding:5px 6px; text-align:right; border-bottom:1px solid var(--border); white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
th { color:var(--muted); font-weight:500; text-transform:uppercase; letter-spacing:.08em; font-size:.58rem; }
td { color:var(--text); }
tr:last-child td { border-bottom:none; }
.ph-back td:not(:first-child){color:var(--back);} .ph-down td:not(:first-child){color:var(--down);}
.ph-contact td:not(:first-child){color:var(--contact);} .ph-follow td:not(:first-child){color:var(--follow);}

.save-row { display:flex; gap:8px; align-items:center; }
#swing-name-input { flex:1; padding:6px 10px; background:var(--surface2); border:1px solid var(--border);
  border-radius:3px; color:var(--text); font-family:'DM Mono',monospace; font-size:.75rem; outline:none; }
#swing-name-input:focus { border-color:var(--accent); }
#btn-save-swing { padding:6px 14px; border:1px solid var(--accent); border-radius:3px;
  background:transparent; color:var(--accent); font-family:'DM Mono',monospace; font-size:.72rem;
  cursor:pointer; transition:all .15s; white-space:nowrap; }
#btn-save-swing:hover { background:var(--accent); color:var(--bg); }
#save-status { font-size:.65rem; color:var(--muted); margin-top:6px; }

/* ── Library ── */
.lib-layout { display:grid; grid-template-columns:280px 1fr; gap:16px; align-items:start; }
.swing-list { display:flex; flex-direction:column; gap:8px; }
.swing-item { background:var(--surface); border:1px solid var(--border); border-radius:5px;
  padding:12px 14px; cursor:pointer; transition:border-color .15s; }
.swing-item:hover { border-color:var(--accent); }
.swing-item.selected { border-color:var(--accent); background:var(--surface2); }
.swing-item.compare-selected { border-color:var(--accent2); }
.swing-name { font-size:.82rem; font-weight:500; color:var(--text); margin-bottom:4px; }
.swing-meta { font-size:.62rem; color:var(--muted); }
.swing-actions { display:flex; gap:6px; margin-top:8px; }
.lib-btn { padding:3px 8px; border-radius:3px; border:1px solid var(--border);
  background:transparent; color:var(--muted); font-family:'DM Mono',monospace; font-size:.62rem;
  cursor:pointer; transition:all .12s; }
.lib-btn:hover { border-color:var(--accent); color:var(--accent); }
.lib-btn.danger:hover { border-color:var(--accent2); color:var(--accent2); }
.lib-btn.compare-btn.active { background:var(--accent2); border-color:var(--accent2); color:var(--bg); }
#lib-empty { color:var(--muted); font-size:.8rem; padding:20px 0; }

.compare-panel { display:flex; flex-direction:column; gap:16px; }
.compare-header { display:flex; align-items:center; justify-content:space-between; }
.compare-videos { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.compare-video-col { display:flex; flex-direction:column; gap:8px; }
.compare-label { font-size:.65rem; color:var(--muted); text-transform:uppercase; letter-spacing:.1em; }
.compare-img-wrap { background:#000; border:1px solid var(--border); border-radius:5px; overflow:hidden; line-height:0; }
.compare-img { width:100%; display:block; }
.compare-scrubber { width:100%; -webkit-appearance:none; height:3px; border-radius:2px; background:var(--border); outline:none; cursor:pointer; }
.compare-scrubber::-webkit-slider-thumb { -webkit-appearance:none; width:11px; height:11px; border-radius:50%; background:var(--accent); cursor:pointer; }

.compare-table-wrap { overflow-x:auto; }
.compare-table { width:100%; border-collapse:collapse; font-size:.68rem; }
.compare-table th, .compare-table td { padding:6px 8px; text-align:right; border-bottom:1px solid var(--border); white-space:nowrap; }
.compare-table th:first-child, .compare-table td:first-child { text-align:left; }
.compare-table th { color:var(--muted); font-weight:500; text-transform:uppercase; letter-spacing:.08em; font-size:.6rem; }
.compare-table td { color:var(--text); }
.compare-table tr:last-child td { border-bottom:none; }
.col-a{color:var(--accent)!important;} .col-b{color:var(--accent2)!important;}
.diff-pos{color:#22dd66!important;} .diff-neg{color:#ff3344!important;} .diff-neu{color:var(--muted)!important;}

#loading { position:fixed; inset:0; background:var(--bg);
  display:flex; flex-direction:column; align-items:center; justify-content:center; gap:14px; z-index:999; }
#loading.hidden { display:none; }
.spinner { width:34px; height:34px; border:3px solid var(--border); border-top-color:var(--accent);
  border-radius:50%; animation:spin .75s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }
#load-msg { font-size:.85rem; color:var(--muted); }
</style>
</head>
<body>

<div id="loading"><div class="spinner"></div><div id="load-msg">Loading swing data...</div></div>

<header>
  <div><div class="logo">SWING<span>TRACK</span></div><div class="tagline">Golf Swing Analyser</div></div>
  <div class="kbd-hints">SPACE &nbsp;play / pause<br>&larr; &rarr; &nbsp;step frame</div>
</header>

<div class="tabs">
  <button class="tab-btn active" data-tab="player">Player</button>
  <button class="tab-btn" data-tab="library">Library</button>
</div>

<!-- PLAYER TAB -->
<div class="tab-panel active" id="tab-player">
<div class="player-layout">
  <div class="video-col">
    <div id="video-wrap"><img id="frame-img" src="" alt="swing frame"></div>
    <div class="controls">
      <div class="ctrl-row">
        <button id="btn-play">&#x25B6;</button>
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
        <button class="phase-jump-btn pj-back"    data-phase="Backswing"      disabled>BACKSWING</button>
        <button class="phase-jump-btn pj-down"    data-phase="Downswing"      disabled>DOWNSWING</button>
        <button class="phase-jump-btn pj-contact" data-phase="Ball Contact"   disabled>BALL CONTACT</button>
        <button class="phase-jump-btn pj-follow"  data-phase="Follow Through" disabled>FOLLOW THRU</button>
      </div>
    </div>
  </div>

  <div class="sidebar">
    <div class="card">
      <div class="section-title">Save to Library</div>
      <div class="save-row">
        <input id="swing-name-input" type="text" placeholder="e.g. Driver Round 4">
        <button id="btn-save-swing">Save</button>
      </div>
      <div id="save-status"></div>
    </div>
    <div class="card">
      <div class="section-title">Current Phase</div>
      <div id="phase-badge">—</div>
    </div>
    <div class="card">
      <div class="section-title">Joint Angles — Current Frame</div>
      <div id="metrics-list"></div>
    </div>
    <div class="card">
      <div class="section-title">Angles at Phase Start</div>
      <div class="phase-table-wrap">
        <table id="phase-table">
          <thead><tr>
            <th>Phase</th><th>L.Arm</th><th>R.Arm</th><th>Sh.Rot</th>
            <th>Hip</th><th>Spine</th><th>L.Kn</th><th>R.Kn</th>
          </tr></thead>
          <tbody id="phase-table-body">
            <tr><td colspan="8" style="color:var(--muted);text-align:center">Loading...</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>
</div>

<!-- LIBRARY TAB -->
<div class="tab-panel" id="tab-library">
<div class="lib-layout">

  <div>
    <div class="card" style="margin-bottom:12px">
      <div class="section-title">Saved Swings</div>
      <div class="swing-list" id="swing-list">
        <div id="lib-empty">No swings saved yet.</div>
      </div>
    </div>
    <div class="card" style="font-size:.65rem;color:var(--muted);line-height:1.9">
      Click a swing to preview.<br>
      Click <span style="color:var(--accent2)">Compare</span> on two<br>swings to compare them.
    </div>
  </div>

  <div>
    <!-- Compare panel -->
    <div class="compare-panel" id="compare-panel" style="display:none">
      <div class="compare-header">
        <div class="section-title" style="margin:0">Comparison</div>
        <button class="lib-btn" id="btn-clear-compare">Clear</button>
      </div>
      <div class="compare-videos">
        <div class="compare-video-col">
          <div class="compare-label col-a" id="compare-label-a">Swing A</div>
          <div class="compare-img-wrap"><img class="compare-img" id="compare-img-a" src="" alt="A"></div>
          <input class="compare-scrubber" id="compare-scrub-a" type="range" min="0" value="0" step="1">
        </div>
        <div class="compare-video-col">
          <div class="compare-label col-b" id="compare-label-b">Swing B</div>
          <div class="compare-img-wrap"><img class="compare-img" id="compare-img-b" src="" alt="B"></div>
          <input class="compare-scrubber" id="compare-scrub-b" type="range" min="0" value="0" step="1">
        </div>
      </div>
      <div class="card">
        <div class="section-title">Angles at Phase Start — Side by Side</div>
        <div class="compare-table-wrap">
          <table class="compare-table">
            <thead><tr>
              <th>Phase</th><th>Metric</th>
              <th id="th-a" class="col-a">A</th>
              <th id="th-b" class="col-b">B</th>
              <th>Diff (B-A)</th>
            </tr></thead>
            <tbody id="compare-table-body"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Single preview -->
    <div id="preview-panel" style="display:none">
      <div class="card">
        <div class="section-title" id="preview-title">Preview</div>
        <div class="compare-img-wrap" style="margin-bottom:10px">
          <img class="compare-img" id="preview-img" src="" alt="preview">
        </div>
        <input class="compare-scrubber" id="preview-scrub" type="range" min="0" value="0" step="1">
      </div>
    </div>
  </div>

</div>
</div>

<script>
const METRIC_LABELS = {
  lead_arm_angle:'Lead Arm', trail_arm_angle:'Trail Arm',
  shoulder_rotation:'Sh.Rot', hip_rotation:'Hip Rot',
  spine_tilt:'Spine', lead_knee_flex:'L.Knee', trail_knee_flex:'R.Knee',
};
const METRIC_KEYS  = Object.keys(METRIC_LABELS);
const PHASE_COLORS = {'Backswing':'pc-back','Downswing':'pc-down','Ball Contact':'pc-contact','Follow Through':'pc-follow','Unknown':'pc-unknown'};
const PHASE_ROW    = {'Backswing':'ph-back','Downswing':'ph-down','Ball Contact':'ph-contact','Follow Through':'ph-follow'};
const PHASE_ORDER  = ['Backswing','Downswing','Ball Contact','Follow Through'];

// ── Player ─────────────────────────────────────────────────────────────
let FRAMES=[], FPS=30, phaseIndex={};
let currentFrame=0, playing=false, speed=1.0, lastTime=null, accumMs=0;

const imgEl=document.getElementById('frame-img'), scrubber=document.getElementById('scrubber'),
  counter=document.getElementById('frame-counter'), btnPlay=document.getElementById('btn-play'),
  phaseBadge=document.getElementById('phase-badge'), metricsList=document.getElementById('metrics-list'),
  loadingEl=document.getElementById('loading'), loadMsg=document.getElementById('load-msg'),
  speedBtns=document.querySelectorAll('.speed-btn'), phaseJumpBtns=document.querySelectorAll('.phase-jump-btn'),
  phaseTableBody=document.getElementById('phase-table-body'),
  customInput=document.getElementById('custom-speed-input'), btnSetSpeed=document.getElementById('btn-set-speed'),
  swingNameInput=document.getElementById('swing-name-input'), btnSaveSwing=document.getElementById('btn-save-swing'),
  saveStatus=document.getElementById('save-status');

function showFrame(idx) {
  currentFrame=Math.max(0,Math.min(idx,FRAMES.length-1));
  const f=FRAMES[currentFrame];
  imgEl.src='data:image/jpeg;base64,'+f.img_b64;
  scrubber.value=currentFrame;
  counter.textContent=currentFrame+' / '+(FRAMES.length-1);
  const phase=f.phase||'Unknown';
  phaseBadge.textContent=phase.toUpperCase();
  phaseBadge.className=PHASE_COLORS[phase]||'pc-unknown';
  phaseJumpBtns.forEach(b=>b.classList.toggle('is-active-phase',b.dataset.phase===phase));
  const m=f.metrics||{};
  metricsList.innerHTML=METRIC_KEYS.map(k=>{
    const v=m[k]!==undefined?m[k].toFixed(1)+'\u00b0':'\u2014';
    return '<div class="metric-row"><span class="metric-name">'+METRIC_LABELS[k]+'</span><span class="metric-value">'+v+'</span></div>';
  }).join('');
}

function tick(ts) {
  if (!playing) return;
  if (lastTime!==null) {
    accumMs+=(ts-lastTime)*speed;
    const mpf=1000/FPS;
    while (accumMs>=mpf) { accumMs-=mpf; if (currentFrame>=FRAMES.length-1){pause();showFrame(0);return;} showFrame(currentFrame+1); }
  }
  lastTime=ts; requestAnimationFrame(tick);
}
function play()  { if (!FRAMES.length) return; playing=true; lastTime=null; accumMs=0; btnPlay.innerHTML='&#x23F8;'; requestAnimationFrame(tick); }
function pause() { playing=false; btnPlay.innerHTML='&#x25B6;'; }

btnPlay.addEventListener('click',()=>playing?pause():play());
scrubber.addEventListener('input',()=>{pause();showFrame(parseInt(scrubber.value));});
speedBtns.forEach(b=>b.addEventListener('click',()=>{
  speed=parseFloat(b.dataset.speed); customInput.value='';
  speedBtns.forEach(x=>x.classList.toggle('active',parseFloat(x.dataset.speed)===speed));
}));
btnSetSpeed.addEventListener('click',()=>{
  const v=parseFloat(customInput.value);
  if (!isNaN(v)&&v>0){speed=v;speedBtns.forEach(b=>b.classList.remove('active'));}
});
customInput.addEventListener('keydown',e=>{if(e.key==='Enter')btnSetSpeed.click();});
phaseJumpBtns.forEach(b=>b.addEventListener('click',()=>{
  const idx=phaseIndex[b.dataset.phase]; if(idx!==undefined){pause();showFrame(idx);}
}));
document.addEventListener('keydown',e=>{
  if (e.target===customInput||e.target===swingNameInput) return;
  if (e.code==='Space'){e.preventDefault();playing?pause():play();}
  if (e.code==='ArrowRight'){pause();showFrame(currentFrame+1);}
  if (e.code==='ArrowLeft'){pause();showFrame(currentFrame-1);}
});

function buildPhaseTable(frames) {
  phaseTableBody.innerHTML=PHASE_ORDER.map(phase=>{
    const cls=PHASE_ROW[phase]||'', idx=phaseIndex[phase];
    const m=idx!==undefined?(frames[idx].metrics||{}):{};
    const cells=METRIC_KEYS.map(k=>m[k]!==undefined?'<td>'+m[k].toFixed(1)+'</td>':'<td>\u2014</td>').join('');
    return '<tr class="'+cls+'"><td>'+phase+'</td>'+cells+'</tr>';
  }).join('');
}
function buildPhaseIndex(frames) {
  frames.forEach((f,i)=>{if(f.phase&&phaseIndex[f.phase]===undefined)phaseIndex[f.phase]=i;});
  phaseJumpBtns.forEach(b=>{if(phaseIndex[b.dataset.phase]!==undefined)b.disabled=false;});
}

// Save
btnSaveSwing.addEventListener('click',async()=>{
  const name=swingNameInput.value.trim();
  if (!name){saveStatus.textContent='Enter a name first.';return;}
  saveStatus.textContent='Saving...'; btnSaveSwing.disabled=true;
  try {
    const r=await fetch('/library/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
    const d=await r.json();
    if (d.ok){saveStatus.style.color='var(--back)';saveStatus.textContent='Saved! (id '+d.id+')';swingNameInput.value='';refreshLibrary();}
    else{saveStatus.style.color='var(--accent2)';saveStatus.textContent='Error: '+d.error;}
  } catch(e){saveStatus.textContent='Failed: '+e;}
  finally{btnSaveSwing.disabled=false;}
});

// Tabs
document.querySelectorAll('.tab-btn').forEach(btn=>{
  btn.addEventListener('click',()=>{
    document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-'+btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab==='library') refreshLibrary();
  });
});

// ── Library ─────────────────────────────────────────────────────────────
let swingsCache={}, swingsList=[];
let selectedId=null, compareIdA=null, compareIdB=null, swingDataCache={};

const swingListEl=document.getElementById('swing-list'), libEmpty=document.getElementById('lib-empty'),
  comparePanel=document.getElementById('compare-panel'), previewPanel=document.getElementById('preview-panel'),
  previewImg=document.getElementById('preview-img'), previewScrub=document.getElementById('preview-scrub'),
  previewTitle=document.getElementById('preview-title'),
  compareImgA=document.getElementById('compare-img-a'), compareImgB=document.getElementById('compare-img-b'),
  compareScrubA=document.getElementById('compare-scrub-a'), compareScrubB=document.getElementById('compare-scrub-b'),
  compareLabelA=document.getElementById('compare-label-a'), compareLabelB=document.getElementById('compare-label-b'),
  compareTableBody=document.getElementById('compare-table-body'),
  thA=document.getElementById('th-a'), thB=document.getElementById('th-b'),
  btnClearCompare=document.getElementById('btn-clear-compare');

async function refreshLibrary() {
  const r=await fetch('/library/list'); swingsList=await r.json();
  swingsList.forEach(s=>swingsCache[s.id]=s);
  renderSwingList();
}

function renderSwingList() {
  swingListEl.innerHTML='';
  if (!swingsList.length){swingListEl.appendChild(libEmpty);return;}
  swingsList.forEach(sw=>{
    const div=document.createElement('div');
    div.className='swing-item';
    if (sw.id===selectedId) div.classList.add('selected');
    if (sw.id===compareIdA||sw.id===compareIdB) div.classList.add('compare-selected');
    const isCompare=sw.id===compareIdA||sw.id===compareIdB;
    div.innerHTML='<div class="swing-name">'+sw.name+'</div>'+
      '<div class="swing-meta">'+sw.created_at.replace('T',' ')+' | '+sw.frame_count+' frames</div>'+
      '<div class="swing-actions">'+
        '<button class="lib-btn compare-btn'+(isCompare?' active':'')+'">Compare</button>'+
        '<button class="lib-btn rename-btn">Rename</button>'+
        '<button class="lib-btn danger delete-btn">Delete</button>'+
      '</div>';
    div.addEventListener('click',e=>{if(e.target.closest('button'))return;selectedId=sw.id;previewSwing(sw.id);renderSwingList();});
    div.querySelector('.compare-btn').addEventListener('click',e=>{
      e.stopPropagation();
      if (compareIdA===sw.id) compareIdA=null;
      else if (compareIdB===sw.id) compareIdB=null;
      else if (!compareIdA) compareIdA=sw.id;
      else if (!compareIdB) compareIdB=sw.id;
      else {compareIdA=sw.id;compareIdB=null;}
      renderSwingList(); updateComparePanel();
    });
    div.querySelector('.rename-btn').addEventListener('click',async e=>{
      e.stopPropagation();
      const n=prompt('New name:',sw.name); if(!n||!n.trim()) return;
      await fetch('/library/rename/'+sw.id,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:n.trim()})});
      refreshLibrary();
    });
    div.querySelector('.delete-btn').addEventListener('click',async e=>{
      e.stopPropagation();
      if (!confirm('Delete "'+sw.name+'"?')) return;
      await fetch('/library/delete/'+sw.id,{method:'DELETE'});
      if (selectedId===sw.id){selectedId=null;previewPanel.style.display='none';}
      if (compareIdA===sw.id) compareIdA=null;
      if (compareIdB===sw.id) compareIdB=null;
      delete swingDataCache[sw.id];
      refreshLibrary(); updateComparePanel();
    });
    swingListEl.appendChild(div);
  });
}

async function loadSwingData(id) {
  if (swingDataCache[id]) return swingDataCache[id];
  const r=await fetch('/library/swing/'+id); const d=await r.json();
  swingDataCache[id]=d; return d;
}

async function previewSwing(id) {
  comparePanel.style.display='none'; previewPanel.style.display='block';
  previewImg.src=''; previewScrub.value=0;
  const sw=swingsCache[id]; previewTitle.textContent=sw?sw.name:'Preview';
  const data=await loadSwingData(id);
  previewScrub.max=data.frames.length-1;
  function show(i){previewImg.src='data:image/jpeg;base64,'+data.frames[Math.max(0,Math.min(i,data.frames.length-1))].img_b64;}
  show(0); previewScrub.oninput=()=>show(parseInt(previewScrub.value));
}

async function updateComparePanel() {
  if (!compareIdA&&!compareIdB){comparePanel.style.display='none';return;}
  if (compareIdA&&!compareIdB){previewSwing(compareIdA);return;}
  if (!compareIdA&&compareIdB){previewSwing(compareIdB);return;}
  previewPanel.style.display='none'; comparePanel.style.display='block';
  const [dataA,dataB]=await Promise.all([loadSwingData(compareIdA),loadSwingData(compareIdB)]);
  const swA=swingsCache[compareIdA], swB=swingsCache[compareIdB];
  compareLabelA.textContent=swA?swA.name:'Swing A';
  compareLabelB.textContent=swB?swB.name:'Swing B';
  thA.textContent=swA?swA.name.substring(0,16):'A';
  thB.textContent=swB?swB.name.substring(0,16):'B';
  compareScrubA.max=dataA.frames.length-1; compareScrubA.value=0;
  compareScrubB.max=dataB.frames.length-1; compareScrubB.value=0;
  function showA(i){compareImgA.src='data:image/jpeg;base64,'+dataA.frames[Math.max(0,Math.min(i,dataA.frames.length-1))].img_b64;}
  function showB(i){compareImgB.src='data:image/jpeg;base64,'+dataB.frames[Math.max(0,Math.min(i,dataB.frames.length-1))].img_b64;}
  showA(0); showB(0);
  compareScrubA.oninput=()=>showA(parseInt(compareScrubA.value));
  compareScrubB.oninput=()=>showB(parseInt(compareScrubB.value));
  const paA=dataA.phase_angles||{}, paB=dataB.phase_angles||{};
  compareTableBody.innerHTML=PHASE_ORDER.flatMap((phase,pi)=>
    METRIC_KEYS.map((k,ki)=>{
      const vA=paA[phase]?.[k], vB=paB[phase]?.[k];
      const aStr=vA!==undefined?vA.toFixed(1):'\u2014';
      const bStr=vB!==undefined?vB.toFixed(1):'\u2014';
      let dStr='\u2014', dCls='diff-neu';
      if (vA!==undefined&&vB!==undefined){
        const d=vB-vA; dStr=(d>0?'+':'')+d.toFixed(1);
        dCls=Math.abs(d)<1?'diff-neu':d>0?'diff-pos':'diff-neg';
      }
      const phaseCell=ki===0?'<td rowspan="'+METRIC_KEYS.length+'" style="vertical-align:middle;border-bottom:1px solid var(--border);color:var(--muted)">'+phase+'</td>':'';
      return '<tr>'+phaseCell+'<td style="color:var(--muted)">'+METRIC_LABELS[k]+'</td>'+
        '<td class="col-a">'+aStr+'</td><td class="col-b">'+bStr+'</td>'+
        '<td class="'+dCls+'">'+dStr+'</td></tr>';
    })
  ).join('');
}

btnClearCompare.addEventListener('click',()=>{
  compareIdA=null; compareIdB=null;
  comparePanel.style.display='none'; previewPanel.style.display='none';
  renderSwingList();
});

// Boot
fetch('/frames').then(r=>r.json()).then(data=>{
  FRAMES=data.frames; FPS=data.fps; scrubber.max=FRAMES.length-1;
  buildPhaseIndex(FRAMES); buildPhaseTable(FRAMES);
  loadingEl.classList.add('hidden'); showFrame(0);
}).catch(err=>{loadMsg.textContent='Error: '+err;});
</script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════════════════
# Flask
# ═══════════════════════════════════════════════════════════════════════

app     = Flask(__name__)
library = SwingLibrary()
_frames_data: list[dict] = []
_fps:         float       = 30.0
_source_name: str         = ""


@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/frames")
def frames():
    return jsonify({"frames": _frames_data, "fps": _fps})

@app.route("/library/save", methods=["POST"])
def lib_save():
    name = request.json.get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "error": "Name is required"})
    try:
        swing_id = library.save_swing(name, _frames_data, _fps, _source_name)
        return jsonify({"ok": True, "id": swing_id})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/library/list")
def lib_list():
    return jsonify(library.list_swings())

@app.route("/library/swing/<int:swing_id>")
def lib_swing(swing_id):
    data = library.load_swing(swing_id)
    if not data:
        return jsonify({"error": "Not found"}), 404
    return jsonify(data)

@app.route("/library/delete/<int:swing_id>", methods=["DELETE"])
def lib_delete(swing_id):
    library.delete_swing(swing_id)
    return jsonify({"ok": True})

@app.route("/library/rename/<int:swing_id>", methods=["POST"])
def lib_rename(swing_id):
    name = request.json.get("name", "").strip()
    if name:
        library.rename_swing(swing_id, name)
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

def main():
    global _frames_data, _fps, _source_name

    parser = argparse.ArgumentParser(description="Golf Swing Web Player")
    parser.add_argument("--source", required=True)
    parser.add_argument("--port", type=int, default=5100)
    args = parser.parse_args()

    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    _source_name = str(source)
    _frames_data, _fps = process_video(source)
    print(f"\n[INFO] Open your browser -> http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()