"""SQLite storage layer. No server to install or configure - the whole
database lives in a single file at data/sentry.db, created automatically
on first run."""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "sentry.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS machines (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    machine_type TEXT NOT NULL,
    location TEXT,
    description TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recordings (
    id TEXT PRIMARY KEY,
    machine_id TEXT NOT NULL REFERENCES machines(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    duration_seconds REAL,
    sample_rate INTEGER,
    is_baseline INTEGER NOT NULL DEFAULT 0,
    features_json TEXT,
    uploaded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS detections (
    id TEXT PRIMARY KEY,
    recording_id TEXT NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    anomaly_score REAL NOT NULL,
    status TEXT NOT NULL,
    model_version TEXT,
    llm_diagnosis_json TEXT,
    created_at TEXT NOT NULL
);
"""


@contextmanager
def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def new_id() -> str:
    return uuid.uuid4().hex


def now() -> str:
    return datetime.utcnow().isoformat()


# ---- machines ----

def create_machine(name: str, machine_type: str, location: str = "", description: str = "") -> str:
    mid = new_id()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO machines (id, name, machine_type, location, description, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (mid, name, machine_type, location, description, now()),
        )
    return mid


def list_machines() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM machines ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_machine(machine_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM machines WHERE id = ?", (machine_id,)).fetchone()
        return dict(row) if row else None


def delete_machine(machine_id: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM machines WHERE id = ?", (machine_id,))


# ---- recordings ----

def create_recording(machine_id: str, filename: str, file_path: str, duration: float,
                      sample_rate: int, is_baseline: bool, features: dict) -> str:
    rid = new_id()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO recordings (id, machine_id, filename, file_path, duration_seconds, "
            "sample_rate, is_baseline, features_json, uploaded_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (rid, machine_id, filename, file_path, duration, sample_rate,
             1 if is_baseline else 0, json.dumps(features), now()),
        )
    return rid


def get_recording(recording_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM recordings WHERE id = ?", (recording_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["features"] = json.loads(d.pop("features_json"))
        return d


def list_recordings(machine_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM recordings WHERE machine_id = ? ORDER BY uploaded_at DESC", (machine_id,)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["features"] = json.loads(d.pop("features_json"))
            out.append(d)
        return out


def baseline_vectors(machine_id: str) -> list[list[float]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT features_json FROM recordings WHERE machine_id = ? AND is_baseline = 1",
            (machine_id,),
        ).fetchall()
        vectors = []
        for r in rows:
            feats = json.loads(r["features_json"])
            if feats and "vector" in feats:
                vectors.append(feats["vector"])
        return vectors


# ---- detections ----

def upsert_detection(recording_id: str, anomaly_score: float, status: str,
                      model_version: str, llm_diagnosis: dict) -> dict:
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM detections WHERE recording_id = ?", (recording_id,)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE detections SET anomaly_score=?, status=?, model_version=?, "
                "llm_diagnosis_json=?, created_at=? WHERE recording_id=?",
                (anomaly_score, status, model_version, json.dumps(llm_diagnosis), now(), recording_id),
            )
            det_id = existing["id"]
        else:
            det_id = new_id()
            conn.execute(
                "INSERT INTO detections (id, recording_id, anomaly_score, status, model_version, "
                "llm_diagnosis_json, created_at) VALUES (?,?,?,?,?,?,?)",
                (det_id, recording_id, anomaly_score, status, model_version,
                 json.dumps(llm_diagnosis), now()),
            )
    return get_detection(det_id)


def get_detection(detection_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (detection_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["llm_diagnosis"] = json.loads(d.pop("llm_diagnosis_json") or "{}")
        return d


def list_detections(machine_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT d.* FROM detections d JOIN recordings r ON d.recording_id = r.id "
            "WHERE r.machine_id = ? ORDER BY d.created_at ASC",
            (machine_id,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["llm_diagnosis"] = json.loads(d.pop("llm_diagnosis_json") or "{}")
            out.append(d)
        return out
