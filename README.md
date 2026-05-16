# ⛳ Golf Swing Tracker

A Python computer-vision tool that analyses golf swings from video using **MediaPipe Pose** and **OpenCV**.

---

## Project Structure

```
golf_swing_tracker/
├── main.py                        # CLI entry point
├── config.yaml                    # Runtime configuration
├── requirements.txt
├── data/
│   ├── raw/                       # Original video files
│   ├── processed/                 # Intermediate outputs
│   └── models/                    # Any custom ML models
├── src/
│   ├── capture/
│   │   └── video_capture.py       # Webcam / file reader
│   ├── analysis/
│   │   ├── pose_estimator.py      # MediaPipe wrapper
│   │   └── swing_metrics.py       # Angle & rotation calculations
│   └── visualization/
│       └── visualizer.py          # Overlay + Matplotlib plots
├── tests/
│   └── test_swing_metrics.py
├── notebooks/                     # Jupyter experimentation
└── outputs/                       # Annotated videos, CSVs, plots
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run on a video file

```bash
python main.py --source data/raw/my_swing.mp4 --save-video --save-csv
```

### 3. Run on live webcam

```bash
python main.py --source 0
# Press Q to quit
```

Outputs land in `outputs/`:

- `annotated_swing.mp4` – video with skeleton + metrics overlay
- `swing_metrics.csv` – per-frame metric values
- `swing_timeline.png` – time-series chart

---

## Metrics Computed

| Metric              | Description                            |
| ------------------- | -------------------------------------- |
| `lead_arm_angle`    | Elbow flex on the lead (left) arm      |
| `trail_arm_angle`   | Elbow flex on the trail (right) arm    |
| `shoulder_rotation` | Shoulder turn relative to camera plane |
| `hip_rotation`      | Hip turn relative to camera plane      |
| `spine_tilt`        | Lateral tilt of the spine              |
| `lead_knee_flex`    | Knee bend, lead leg                    |
| `trail_knee_flex`   | Knee bend, trail leg                   |

All values are in **degrees**.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Next Steps

- [ ] Automatic swing phase detection (address → backswing → impact → follow-through)
- [ ] Tempo & rhythm scoring
- [ ] Side-by-side comparison of two swings
- [ ] Web dashboard (Streamlit or FastAPI)
- [ ] Rep-over-rep trend tracking
