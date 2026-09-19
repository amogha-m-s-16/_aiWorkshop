# Sentry — AI Machine Sound Detector (Streamlit edition)

The same idea as a full React/FastAPI/Postgres app, rebuilt on a stack
that's much faster to get running: **one Python process, one command,
no Docker, no database server to install.**

- **UI + backend**: [Streamlit](https://streamlit.io) — a single Python
  app, no separate frontend build step.
- **Database**: SQLite — a plain file at `data/sentry.db`, created
  automatically. No server, no credentials.
- **Feature extraction**: `librosa` (MFCCs, spectral stats, RMS, zero-crossing rate).
- **Anomaly detection**: `scikit-learn` `IsolationForest`, trained per
  machine on its own "baseline" (known-healthy) recordings.
- **Diagnosis**: Claude (Anthropic API) turns the anomaly score + spectral
  stats into a plain-language report — probable causes, recommended
  actions, confidence. Falls back to a rule-based report if no API key is
  configured, so the app still works without one.

## Why this is easier to use than the React/FastAPI/Docker version

| | This version | React/FastAPI/Postgres version |
|---|---|---|
| Setup | `pip install -r requirements.txt` | separate frontend + backend installs, or Docker |
| Run | `streamlit run app.py` | run two services (or `docker compose up`) |
| Database | SQLite file, zero config | PostgreSQL server required |
| Processes | 1 | 2–4 (frontend, backend, db, optionally nginx) |

Trade-off: Streamlit's UI is less customizable than a hand-built React
frontend, and it's a single-user-at-a-time app by default rather than a
production multi-tenant service. For prototyping, a shop floor, or a small
team, that's usually a good trade.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY (optional - app works without it)
streamlit run app.py
```

Streamlit opens the app in your browser automatically (usually at
http://localhost:8501).

## Using the app

1. **Add a machine** in the sidebar (name, type, optional location/notes).
2. **Upload a few "baseline" recordings** — audio of the machine running
   normally — with the baseline checkbox checked. Once a machine has 5+
   baseline samples, a dedicated anomaly model trains automatically (you
   can also trigger this from the Dashboard tab).
3. **Upload a regular recording** (checkbox unchecked) to score it. You'll
   get an anomaly score (0–100) on a gauge, a status (normal / warning /
   critical), and a Claude-written diagnostic report.
4. Check the **Dashboard tab** for the anomaly-score trend over time and
   the full recordings table.

## Project layout

```
app.py              Streamlit UI and page flow
db.py                SQLite schema + CRUD helpers
audio_features.py    librosa feature extraction
anomaly_model.py     per-machine IsolationForest training/scoring
llm_diagnosis.py     Claude-based diagnostic report generation
.streamlit/config.toml   dark theme matching the acoustic-monitoring look
data/                SQLite db, uploaded audio, trained models (gitignored)
```

## Notes on the ML approach

- `IsolationForest` is unsupervised and needs no labeled "fault" data —
  appropriate here since you rarely have recordings of every failure mode
  but do have plenty of normal-operation audio.
- Anomaly scores are normalized to 0–100. Thresholds (normal < 40,
  warning < 70, critical ≥ 70) live in `anomaly_model.py` and are easy to
  tune per deployment.
- This is a reasonable starting point, not a certified predictive-maintenance
  system — validate thresholds and the feature set against real recordings
  from your machines before relying on it operationally.

## Deploying beyond your own machine

Streamlit apps deploy easily to [Streamlit Community Cloud](https://streamlit.io/cloud)
or any host that can run a long-lived Python process (a small VM, Render,
Railway, etc.) — just make sure `ANTHROPIC_API_KEY` is set as an
environment variable there, and that `data/` is on persistent storage if
you want recordings and trained models to survive restarts.
