# filename: streamlit_app.py
# purpose:  SmartVision AI — Streamlit entry point (Home page).
#           Shows API health status and champion model stats loaded from artifacts.
#           All inference pages are in pages/ — this page makes no model calls.

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# ── Path setup (run from project root) ────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from streamlit_app import api_client

COMPARISON_PATH = config.ARTIFACTS_DIR / "comparison" / "model_metrics.json"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartVision AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Champion stats (loaded from artifact, not hardcoded) ──────────────────────
@st.cache_data(ttl=3600)
def load_champion_stats() -> dict:
    try:
        m = json.loads(COMPARISON_PATH.read_text())
        champion = m["champion_classifier"]
        return {
            "champion":  champion,
            "acc":       m["champion_test_accuracy"],
            "yolo_map":  m["detection"]["yolov8n"]["map50"],
            "acc_str":   f"{m['champion_test_accuracy']:.1%}",
            "map_str":   f"{m['detection']['yolov8n']['map50']:.1%}",
            "lat_str":   f"{m['models'][champion]['cpu_inference_ms']:.0f} ms",
        }
    except (FileNotFoundError, KeyError):
        return {
            "champion": "resnet50", "acc": 0.655, "yolo_map": 0.147,
            "acc_str": "65.5%", "map_str": "14.7%", "lat_str": "115 ms",
        }


# ── Header ────────────────────────────────────────────────────────────────────
st.title("SmartVision AI")
st.markdown(
    "**Intelligent Multi-Class Object Recognition System** — "
    "4 CNN classifiers + YOLOv8 detection, "
    "served via FastAPI with Redis caching and Prometheus drift monitoring."
)
st.divider()

# ── API health ────────────────────────────────────────────────────────────────
col_health, col_gap = st.columns([2, 3])

with col_health:
    st.subheader("API Status")
    try:
        health = api_client.get_health()
        if health.get("models_ready"):
            loaded = ", ".join(health.get("models_loaded", []))
            st.success(f"All models loaded ({loaded})")
        else:
            loaded = ", ".join(health.get("models_loaded", []))
            st.warning(f"Models loading… ({loaded or 'none yet'})")
    except RuntimeError:
        if api_client.is_hf_spaces():
            st.info("Demo mode — FastAPI not running on HF Spaces free tier.")
        else:
            st.warning("FastAPI offline. Run: `uvicorn api.main:app --reload`")

# ── Key metrics ───────────────────────────────────────────────────────────────
st.subheader("Project at a Glance")
stats = load_champion_stats()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Classes", config.NUM_CLASSES)
c2.metric("CNN Models", 4)
c3.metric(f"Champion ({stats['champion'].upper()})", stats["acc_str"])
c4.metric("YOLOv8n mAP50", stats["map_str"])
c5.metric(f"{stats['champion'].upper()} Latency (CPU)", stats["lat_str"])

st.divider()

# ── Navigation guide ──────────────────────────────────────────────────────────
st.subheader("Pages")
nav = {
    "Classify":          "Upload an image → ResNet50 or MobileNet top-5 predictions with confidence bar chart.",
    "Detect":            "Upload an image → YOLOv8n bounding boxes overlaid on the original photo.",
    "Model Comparison":  "Accuracy, speed vs accuracy scatter, YOLO per-class AP50, full metrics table.",
    "Drift Monitor":     "Per-class KS drift status vs MobileNet val-split baseline. Click Refresh to update.",
    "EDA Insights":      "Dataset overview, image quality stats, detection density — no API required.",
}
for page, desc in nav.items():
    st.markdown(f"**{page}** — {desc}")

st.divider()
st.caption(
    "Dataset: COCO 2017 subset · 22 classes · 200 images/class · "
    "ResNet50 65.5% test accuracy · YOLOv8n mAP50 14.7% @ 50 epochs"
)
