"""Sentry - AI Machine Sound Detector (Streamlit edition)

Run with:  streamlit run app.py

Everything runs in a single process: no separate backend server, no
database server to install (SQLite is a plain file in data/sentry.db), and
no build step for a frontend.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

import anomaly_model
import audio_features
import db
import llm_diagnosis

load_dotenv()

UPLOAD_DIR = Path(__file__).parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

STATUS_COLOR = {"normal": "#4FA57A", "warning": "#F2A93B", "critical": "#D6564A", "unknown": "#6E8494"}

st.set_page_config(page_title="Sentry \u00b7 Machine Sound Detector", page_icon="\U0001F50A", layout="wide")
db.init_db()

# ---------------------------------------------------------------- styling
st.markdown(
    """
    <style>
    .stApp { font-family: 'Inter', sans-serif; }
    div[data-testid="stMetric"] {
        background-color: #1B2127; border: 1px solid #2E3944;
        padding: 12px 16px; border-radius: 2px;
    }
    .status-pill {
        display: inline-block; padding: 3px 10px; border-radius: 2px;
        font-size: 12px; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def status_pill(status: str) -> str:
    color = STATUS_COLOR.get(status, STATUS_COLOR["unknown"])
    return f'<span class="status-pill" style="background:{color}22;color:{color};border:1px solid {color}">{status}</span>'


# ---------------------------------------------------------------- sidebar: machine management
st.sidebar.title("\U0001F50A Sentry")
st.sidebar.caption("Acoustic condition monitoring")

if not os.environ.get("ANTHROPIC_API_KEY"):
    st.sidebar.warning(
        "No ANTHROPIC_API_KEY set. Diagnoses will use a rule-based fallback "
        "instead of Claude. See .env.example.",
        icon="\u26a0\ufe0f",
    )

machines = db.list_machines()
machine_options = {m["name"]: m["id"] for m in machines}

if "selected_machine_id" not in st.session_state and machines:
    st.session_state.selected_machine_id = machines[0]["id"]

if machines:
    names = list(machine_options.keys())
    current_name = next((n for n, i in machine_options.items() if i == st.session_state.get("selected_machine_id")), names[0])
    chosen = st.sidebar.radio("Machines", names, index=names.index(current_name), label_visibility="collapsed")
    st.session_state.selected_machine_id = machine_options[chosen]
else:
    st.sidebar.info("No machines yet - add one below to get started.")

with st.sidebar.expander("+ Add machine", expanded=not machines):
    with st.form("add_machine_form", clear_on_submit=True):
        new_name = st.text_input("Name")
        new_type = st.text_input("Type (e.g. CNC Lathe, HVAC Compressor)")
        new_location = st.text_input("Location (optional)")
        new_desc = st.text_area("Notes (optional)", height=60)
        submitted = st.form_submit_button("Add machine")
        if submitted and new_name and new_type:
            mid = db.create_machine(new_name, new_type, new_location, new_desc)
            st.session_state.selected_machine_id = mid
            st.rerun()

if machines:
    with st.sidebar.expander("Danger zone"):
        if st.button("Delete this machine", type="secondary"):
            db.delete_machine(st.session_state.selected_machine_id)
            st.session_state.pop("selected_machine_id", None)
            st.rerun()

# ---------------------------------------------------------------- main content
if not machines:
    st.title("Welcome to Sentry")
    st.write(
        "Add your first machine in the sidebar, then upload a few 'baseline' "
        "recordings of it running normally, followed by test recordings to "
        "check for acoustic anomalies."
    )
    st.stop()

machine = db.get_machine(st.session_state.selected_machine_id)
recordings = db.list_recordings(machine["id"])
detections = db.list_detections(machine["id"])
baseline_count = sum(1 for r in recordings if r["is_baseline"])
latest_detection = detections[-1] if detections else None

header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title(machine["name"])
    subtitle = machine["machine_type"]
    if machine["location"]:
        subtitle += f" \u00b7 {machine['location']}"
    st.caption(subtitle)
with header_col2:
    st.markdown(
        f"<div style='text-align:right;padding-top:1.2em'>{status_pill(latest_detection['status'] if latest_detection else 'unknown')}</div>",
        unsafe_allow_html=True,
    )

m1, m2, m3, m4 = st.columns(4)
m1.metric("Recordings", len(recordings))
m2.metric("Baseline samples", baseline_count)
m3.metric("Latest score", f"{latest_detection['anomaly_score']:.0f}" if latest_detection else "\u2014")
avg_score = round(sum(d["anomaly_score"] for d in detections) / len(detections), 1) if detections else None
m4.metric("Average score", f"{avg_score:.0f}" if avg_score is not None else "\u2014")

tab_analyze, tab_dashboard = st.tabs(["\U0001F3A7 Analyze", "\U0001F4CA Dashboard"])

# ---------------------------------------------------------------- Analyze tab
with tab_analyze:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Submit a recording")
        st.write(
            "Mark a recording as **baseline** when the machine is known to be "
            "running healthily - the detector learns each machine's normal "
            "sound from these. Everything else gets scored against it."
        )
        uploaded_file = st.file_uploader(
            "Audio file", type=["wav", "mp3", "flac", "ogg", "m4a"], label_visibility="collapsed"
        )
        if uploaded_file:
            st.audio(uploaded_file)
        is_baseline = st.checkbox("This recording is a known-healthy baseline sample")
        analyze_clicked = st.button(
            "Add baseline sample" if is_baseline else "Analyze sound",
            type="primary",
            disabled=uploaded_file is None,
            use_container_width=True,
        )

        if analyze_clicked and uploaded_file is not None:
            ext = Path(uploaded_file.name).suffix.lower()
            dest = UPLOAD_DIR / f"{db.new_id()}{ext}"
            dest.write_bytes(uploaded_file.getvalue())

            with st.spinner("Extracting acoustic features\u2026"):
                try:
                    extracted = audio_features.extract_features(str(dest))
                except Exception as e:
                    st.error(f"Could not process audio file: {e}")
                    st.stop()

            rid = db.create_recording(
                machine_id=machine["id"], filename=uploaded_file.name, file_path=str(dest),
                duration=extracted["duration_seconds"], sample_rate=extracted["sample_rate"],
                is_baseline=is_baseline, features=extracted,
            )

            if is_baseline:
                anomaly_model.train_machine_model(machine["id"], db.baseline_vectors(machine["id"]))
                st.success("Baseline sample saved.")
                st.session_state.last_result = None
            else:
                with st.spinner("Scoring against machine's acoustic baseline\u2026"):
                    vector = extracted["vector"]
                    anomaly_score, model_version = anomaly_model.score(machine["id"], vector)
                    status = anomaly_model.status_from_score(anomaly_score)
                with st.spinner("Generating diagnostic report\u2026"):
                    diagnosis = llm_diagnosis.generate_diagnosis(
                        machine_name=machine["name"], machine_type=machine["machine_type"],
                        anomaly_score=anomaly_score, status=status, stats=extracted["stats"],
                        is_first_recording=baseline_count < anomaly_model.MIN_BASELINE_SAMPLES,
                    )
                detection = db.upsert_detection(rid, anomaly_score, status, model_version, diagnosis)
                st.session_state.last_result = detection
            st.rerun()

    with right:
        st.subheader("Detection result")
        result = st.session_state.get("last_result")
        if not result:
            st.info(
                "Analysis output will appear here once a non-baseline "
                "recording has been submitted for this machine."
            )
        else:
            diag = result["llm_diagnosis"] or {}
            gauge_col, text_col = st.columns([1, 1])
            with gauge_col:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["anomaly_score"],
                    number={"suffix": " / 100", "font": {"size": 28}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": STATUS_COLOR.get(result["status"], "#8FA3AE")},
                        "steps": [
                            {"range": [0, 40], "color": "#1B2127"},
                            {"range": [40, 70], "color": "#232B1E"},
                            {"range": [70, 100], "color": "#2B1B1B"},
                        ],
                    },
                ))
                fig.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=10),
                                   paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E7EAEC"})
                st.plotly_chart(fig, use_container_width=True)
            with text_col:
                st.markdown(status_pill(result["status"]), unsafe_allow_html=True)
                st.write(diag.get("summary", ""))
                if diag.get("severity_explanation"):
                    st.caption(diag["severity_explanation"])

            if diag.get("probable_causes"):
                st.markdown("**Probable causes**")
                for c in diag["probable_causes"]:
                    st.markdown(f"- {c}")
            if diag.get("recommended_actions"):
                st.markdown("**Recommended actions**")
                for a in diag["recommended_actions"]:
                    st.markdown(f"- {a}")
            st.caption(f"Confidence: {diag.get('confidence', 'n/a')} \u00b7 Model: {result['model_version']}")

# ---------------------------------------------------------------- Dashboard tab
with tab_dashboard:
    st.subheader("Anomaly score history")
    if detections:
        df = pd.DataFrame(
            [{"time": d["created_at"], "score": d["anomaly_score"], "status": d["status"]} for d in detections]
        )
        df["time"] = pd.to_datetime(df["time"])
        st.line_chart(df.set_index("time")["score"], height=280)
    else:
        st.caption("No detections logged for this machine yet.")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("Retrain model on baselines", use_container_width=True):
            vectors = db.baseline_vectors(machine["id"])
            trained = anomaly_model.train_machine_model(machine["id"], vectors)
            if trained:
                st.success(f"Model refit on {len(vectors)} baseline samples.")
            else:
                st.warning(
                    f"Need {anomaly_model.MIN_BASELINE_SAMPLES} baseline samples to train a "
                    f"dedicated model (have {len(vectors)}). Using the generic bootstrap model "
                    "until then."
                )

    st.subheader("Recordings")
    if recordings:
        table = pd.DataFrame([
            {
                "File": r["filename"],
                "Duration": f"{r['duration_seconds']:.1f}s" if r["duration_seconds"] else "\u2014",
                "Sample rate": f"{r['sample_rate']} Hz" if r["sample_rate"] else "\u2014",
                "Type": "baseline" if r["is_baseline"] else "sample",
                "Uploaded": r["uploaded_at"][:19].replace("T", " "),
            }
            for r in recordings
        ])
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.caption("No recordings uploaded yet.")
