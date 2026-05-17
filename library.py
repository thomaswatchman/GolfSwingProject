"""
library.py
SQLite-backed swing library.

Stores:
  - Swing metadata (name, date, source filename, fps, frame count)
  - Per-phase instantaneous angles
  - All processed frames as compressed JPEG blobs

Usage (from player.py):
    from library import SwingLibrary
    lib = SwingLibrary()
    swing_id = lib.save_swing("Driver - Round 1", frames_data, fps, "GX016551.MP4")
    swings    = lib.list_swings()
    swing     = lib.load_swing(swing_id)
    lib.delete_swing(swing_id)
"""

from __future__ import annotations

import json
import sqlite3
import zlib
import base64
from datetime import datetime
from pathlib import Path


DB_PATH = Path("swing_library.db")

PHASE_ORDER = ["Backswing", "Downswing", "Ball Contact", "Follow Through"]
METRIC_KEYS = [
    "lead_arm_angle", "trail_arm_angle", "shoulder_rotation",
    "hip_rotation", "spine_tilt", "lead_knee_flex", "trail_knee_flex",
]


class SwingLibrary:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS swings (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT NOT NULL,
                    source      TEXT,
                    fps         REAL NOT NULL,
                    frame_count INTEGER NOT NULL,
                    created_at  TEXT NOT NULL,
                    phase_angles TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS frames (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    swing_id INTEGER NOT NULL REFERENCES swings(id) ON DELETE CASCADE,
                    frame_no INTEGER NOT NULL,
                    phase    TEXT NOT NULL,
                    metrics  TEXT NOT NULL,
                    img_blob BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_frames_swing
                    ON frames(swing_id, frame_no);
            """)

    # ------------------------------------------------------------------
    def save_swing(
        self,
        name: str,
        frames_data: list[dict],
        fps: float,
        source: str = "",
    ) -> int:
        """
        Persist a processed swing. Returns the new swing ID.
        frames_data is the list produced by process_video() in player.py.
        """
        # Build phase_angles: first-frame metrics for each phase
        phase_index: dict[str, int] = {}
        for i, f in enumerate(frames_data):
            p = f.get("phase", "")
            if p and p not in phase_index:
                phase_index[p] = i

        phase_angles: dict[str, dict] = {}
        for phase in PHASE_ORDER:
            idx = phase_index.get(phase)
            if idx is not None:
                phase_angles[phase] = frames_data[idx].get("metrics", {})

        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO swings (name, source, fps, frame_count, created_at, phase_angles)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (name, source, fps, len(frames_data),
                 datetime.now().isoformat(timespec="seconds"),
                 json.dumps(phase_angles)),
            )
            swing_id = cur.lastrowid

            # Store frames in batches
            batch = []
            for i, f in enumerate(frames_data):
                # Compress the base64 image bytes
                raw  = base64.b64decode(f["img_b64"])
                blob = zlib.compress(raw, level=3)
                batch.append((
                    swing_id, i,
                    f.get("phase", ""),
                    json.dumps(f.get("metrics", {})),
                    blob,
                ))
                if len(batch) >= 200:
                    conn.executemany(
                        "INSERT INTO frames (swing_id,frame_no,phase,metrics,img_blob) VALUES (?,?,?,?,?)",
                        batch,
                    )
                    batch.clear()

            if batch:
                conn.executemany(
                    "INSERT INTO frames (swing_id,frame_no,phase,metrics,img_blob) VALUES (?,?,?,?,?)",
                    batch,
                )

        print(f"[LIB] Saved swing '{name}' (id={swing_id}, {len(frames_data)} frames)")
        return swing_id

    # ------------------------------------------------------------------
    def list_swings(self) -> list[dict]:
        """Return summary rows for all saved swings (no frame data)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, source, fps, frame_count, created_at, phase_angles "
                "FROM swings ORDER BY created_at DESC"
            ).fetchall()
        return [
            {
                "id":           r["id"],
                "name":         r["name"],
                "source":       r["source"],
                "fps":          r["fps"],
                "frame_count":  r["frame_count"],
                "created_at":   r["created_at"],
                "phase_angles": json.loads(r["phase_angles"]),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    def load_swing(self, swing_id: int) -> dict | None:
        """Load a full swing including all frames. Returns None if not found."""
        with self._connect() as conn:
            meta = conn.execute(
                "SELECT * FROM swings WHERE id=?", (swing_id,)
            ).fetchone()
            if not meta:
                return None

            rows = conn.execute(
                "SELECT frame_no, phase, metrics, img_blob "
                "FROM frames WHERE swing_id=? ORDER BY frame_no",
                (swing_id,),
            ).fetchall()

        frames = []
        for r in rows:
            raw    = zlib.decompress(r["img_blob"])
            img_b64 = base64.b64encode(raw).decode()
            frames.append({
                "img_b64": img_b64,
                "phase":   r["phase"],
                "metrics": json.loads(r["metrics"]),
            })

        return {
            "id":           meta["id"],
            "name":         meta["name"],
            "source":       meta["source"],
            "fps":          meta["fps"],
            "frame_count":  meta["frame_count"],
            "created_at":   meta["created_at"],
            "phase_angles": json.loads(meta["phase_angles"]),
            "frames":       frames,
        }

    # ------------------------------------------------------------------
    def delete_swing(self, swing_id: int):
        with self._connect() as conn:
            conn.execute("DELETE FROM swings WHERE id=?", (swing_id,))
        print(f"[LIB] Deleted swing id={swing_id}")

    # ------------------------------------------------------------------
    def rename_swing(self, swing_id: int, new_name: str):
        with self._connect() as conn:
            conn.execute("UPDATE swings SET name=? WHERE id=?", (new_name, swing_id))