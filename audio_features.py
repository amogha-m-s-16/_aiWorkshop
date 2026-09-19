"""Extracts a fixed-length numeric feature vector from a raw audio file.

MFCCs capture overall timbre, spectral centroid/bandwidth/rolloff capture
'brightness' and energy distribution, zero-crossing rate captures
noisiness/friction, and RMS captures loudness. Together these are
sensitive to the kinds of changes (bearing wear, belt slip, imbalance,
cavitation, etc.) that show up as shifts in a machine's acoustic signature.
"""
from __future__ import annotations

import numpy as np
import librosa

N_MFCC = 20
SAMPLE_RATE = 22050


def extract_features(file_path: str) -> dict:
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    if y.size == 0:
        raise ValueError("Audio file is empty or unreadable")

    duration = float(librosa.get_duration(y=y, sr=sr))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]

    stats = {
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_centroid_std": float(np.std(spectral_centroid)),
        "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
        "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
        "zero_crossing_rate_mean": float(np.mean(zcr)),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
    }

    vector = np.concatenate([
        mfcc_mean, mfcc_std,
        [stats["spectral_centroid_mean"]],
        [stats["spectral_centroid_std"]],
        [stats["spectral_bandwidth_mean"]],
        [stats["spectral_rolloff_mean"]],
        [stats["zero_crossing_rate_mean"]],
        [stats["rms_mean"]],
        [stats["rms_std"]],
    ]).astype(float)

    return {
        "vector": vector.tolist(),
        "duration_seconds": duration,
        "sample_rate": sr,
        "stats": stats,
    }
