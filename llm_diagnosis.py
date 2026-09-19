"""Turns raw anomaly-detection output into a human-readable diagnostic
report using Claude. The acoustic model produces the score; the LLM's job
is to translate that score plus spectral statistics into plain language a
maintenance technician can act on."""
from __future__ import annotations

import json
import logging
import os

from anthropic import Anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an assistant embedded in an industrial acoustic \
condition-monitoring system. You are given: (1) metadata about a machine, \
(2) an anomaly score from an unsupervised acoustic anomaly detector \
(0 = sounds identical to this machine's healthy baseline, 100 = extremely \
different from baseline), and (3) summary spectral statistics of the \
recording. Write a short, practical diagnostic report for a maintenance \
technician.

Respond with ONLY a JSON object, no prose before or after, no markdown \
fences, matching exactly this shape:
{
  "summary": "one or two sentence plain-language summary",
  "probable_causes": ["short phrase", "short phrase"],
  "recommended_actions": ["short imperative action", "short imperative action"],
  "confidence": "low" | "medium" | "high",
  "severity_explanation": "one sentence on why this score maps to this severity"
}

Guidance:
- If the anomaly score is low (under 40), the summary should say the sound \
is consistent with normal operation - do not invent problems.
- probable_causes should be grounded in the spectral stats given (e.g. a \
sharp rise in zero-crossing rate and high-frequency energy suggests \
friction/grinding; strong low-frequency energy growth can suggest \
imbalance or a loose mount; irregular RMS variance can suggest \
intermittent knocking). Never state a cause as certain - these are \
hypotheses for a technician to check.
- confidence should reflect how much the stats actually support a specific \
cause versus a generic "elevated anomaly" reading.
- Keep each list to at most 4 items. Keep the whole report concise."""


def _fallback_diagnosis(anomaly_score: float, status: str) -> dict:
    if status == "normal":
        summary = "Acoustic signature is consistent with this machine's healthy baseline."
        causes, actions = [], ["No action needed.", "Continue routine monitoring."]
    elif status == "warning":
        summary = "Acoustic signature shows a moderate deviation from the healthy baseline."
        causes = ["Early-stage wear", "Minor load or airflow change", "Sensor placement drift"]
        actions = ["Schedule a closer inspection", "Compare against the most recent baseline recording"]
    else:
        summary = "Acoustic signature shows a large deviation from the healthy baseline."
        causes = ["Bearing or gear wear", "Loose or misaligned component", "Developing mechanical fault"]
        actions = ["Inspect the machine as soon as possible", "Take a fresh baseline once resolved"]
    return {
        "summary": summary + " (LLM diagnosis unavailable - showing rule-based fallback.)",
        "probable_causes": causes,
        "recommended_actions": actions,
        "confidence": "low",
        "severity_explanation": f"Anomaly score {anomaly_score}/100 maps to '{status}' by fixed thresholds.",
    }


def generate_diagnosis(machine_name: str, machine_type: str, anomaly_score: float,
                        status: str, stats: dict, is_first_recording: bool = False) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set - using rule-based fallback diagnosis")
        return _fallback_diagnosis(anomaly_score, status)

    user_content = json.dumps({
        "machine_name": machine_name,
        "machine_type": machine_type,
        "anomaly_score_0_100": anomaly_score,
        "status": status,
        "spectral_stats": stats,
        "note": "No baseline model existed yet for this machine; using a generic bootstrap model."
        if is_first_recording else None,
    })

    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(text)
        parsed.setdefault("confidence", "medium")
        return parsed
    except Exception:
        logger.exception("LLM diagnosis generation failed, using fallback")
        return _fallback_diagnosis(anomaly_score, status)
