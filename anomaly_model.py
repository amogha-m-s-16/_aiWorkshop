"""Per-machine anomaly detection using an IsolationForest trained on that
machine's own 'baseline' (known-healthy) recordings - a healthy compressor
and a healthy lathe simply sound different, so there's no universal
'normal'. Until a machine has enough baseline recordings, a generic
bootstrap model keeps the app usable immediately after setup."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).parent / "data" / "models"
MODEL_VERSION = "isoforest-v1"
MIN_BASELINE_SAMPLES = 5


def _paths(machine_id: str) -> tuple[Path, Path]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR / f"{machine_id}_iforest.joblib", MODEL_DIR / f"{machine_id}_scaler.joblib"


def _bootstrap_paths() -> tuple[Path, Path]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR / "_bootstrap_iforest.joblib", MODEL_DIR / "_bootstrap_scaler.joblib"


def _fit_and_save(vectors: list[list[float]], model_path: Path, scaler_path: Path):
    X = np.array(vectors, dtype=float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = IsolationForest(n_estimators=100, contamination=0.1, random_state=42).fit(Xs)
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)


def ensure_bootstrap_model(feature_dim: int):
    model_path, scaler_path = _bootstrap_paths()
    if model_path.exists() and scaler_path.exists():
        return
    rng = np.random.default_rng(42)
    synthetic = rng.normal(loc=0.0, scale=1.0, size=(200, feature_dim))
    _fit_and_save(synthetic.tolist(), model_path, scaler_path)


def train_machine_model(machine_id: str, baseline_vectors: list[list[float]]) -> bool:
    """Returns True if a dedicated model now exists for this machine."""
    if len(baseline_vectors) < MIN_BASELINE_SAMPLES:
        return False
    model_path, scaler_path = _paths(machine_id)
    _fit_and_save(baseline_vectors, model_path, scaler_path)
    return True


def score(machine_id: str, vector: list[float]) -> tuple[float, str]:
    """Returns (normalized_anomaly_score 0-100, model_version_used)."""
    model_path, scaler_path = _paths(machine_id)
    used_version = MODEL_VERSION

    if model_path.exists() and scaler_path.exists():
        clf: IsolationForest = joblib.load(model_path)
        scaler: StandardScaler = joblib.load(scaler_path)
    else:
        ensure_bootstrap_model(len(vector))
        bmodel_path, bscaler_path = _bootstrap_paths()
        clf = joblib.load(bmodel_path)
        scaler = joblib.load(bscaler_path)
        used_version = f"{MODEL_VERSION}-bootstrap"

    X = np.array([vector], dtype=float)
    Xs = scaler.transform(X)
    raw = clf.decision_function(Xs)[0]  # higher raw = more normal
    anomaly = -raw
    normalized = float(np.clip((anomaly + 0.5) * 100, 0, 100))
    return round(normalized, 2), used_version


def status_from_score(anomaly_score: float) -> str:
    if anomaly_score < 40:
        return "normal"
    if anomaly_score < 70:
        return "warning"
    return "critical"
