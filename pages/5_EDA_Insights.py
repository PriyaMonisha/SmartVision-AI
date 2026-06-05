# filename: pages/5_EDA_Insights.py
# purpose:  Dataset analysis from eda_summary.json — no API calls.
#           Guards against missing file. All values from Section 3 EDA run.

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config

EDA_PATH = config.ARTIFACTS_DIR / "eda" / "eda_summary.json"

st.set_page_config(page_title="EDA Insights — SmartVision AI", layout="wide")
st.title("EDA Insights")
st.caption("Pre-computed from Section 3 EDA notebook. No FastAPI required.")

# ── Load EDA (guard against missing file) ─────────────────────────────────────
@st.cache_data
def load_eda() -> dict:
    if not EDA_PATH.exists():
        return {}
    return json.loads(EDA_PATH.read_text())


eda = load_eda()
if not eda:
    st.warning(
        f"EDA summary not found at `{EDA_PATH}`. "
        "Run `notebooks/02_eda.py` to generate it."
    )
    st.stop()

# ── Section 1: Dataset overview ───────────────────────────────────────────────
st.subheader("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("EDA sample size",   eda.get("total_images", "—"))
col2.metric("Classes in EDA",    eda.get("num_classes", "—"))
col3.metric("Active classes",    config.NUM_CLASSES,
            help="25 original − 3 removed (stop sign, cow, elephant — acquisition failed)")
col4.metric("Images per class",  config.IMAGES_PER_CLASS,
            help="Full dataset: 200 img/class (2nd acquisition round)")

# Class balance chi-squared test
balance = eda.get("class_balance", {})
if balance:
    balanced = balance.get("balanced", True)
    chi2     = balance.get("chi2", 0.0)
    p_val    = balance.get("p_value", 1.0)
    if balanced:
        st.success(f"Class balance: chi2={chi2:.4f}, p={p_val:.4f} — perfectly balanced (10/class in EDA sample).")
    else:
        st.warning(f"Class imbalance detected: chi2={chi2:.4f}, p={p_val:.4f}.")

st.caption(
    "Note: 3 classes removed after acquisition: **stop sign**, **cow**, **elephant** — "
    "HuggingFace COCO streaming returned 0 images for these categories."
)

# ── Split counts bar chart ────────────────────────────────────────────────────
split_counts = eda.get("split_counts", {})
if split_counts:
    with st.expander("EDA split counts per class"):
        train_counts = split_counts.get("train", {})
        val_counts   = split_counts.get("val", {})
        test_counts  = split_counts.get("test", {})

        classes_shown = list(train_counts.keys())
        fig = go.Figure()
        for split_name, counts in [("Train", train_counts), ("Val", val_counts), ("Test", test_counts)]:
            fig.add_trace(go.Bar(
                name=split_name,
                x=classes_shown,
                y=[counts.get(c, 0) for c in classes_shown],
            ))
        fig.update_layout(
            barmode="stack",
            title="EDA Sample Split Counts per Class",
            xaxis=dict(tickangle=-45),
            height=350,
            margin=dict(l=20, r=20, t=50, b=80),
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Section 2: Image quality ──────────────────────────────────────────────────
st.subheader("Image Quality")

iq = eda.get("image_quality", {})
if iq:
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Mean brightness",     f"{iq.get('mean_brightness', 0):.1f} / 255")
    q2.metric("Mean contrast (std)", f"{iq.get('mean_contrast', 0):.1f}")
    q3.metric("Low brightness",      iq.get("low_brightness_count", 0),
              help="< 30 mean brightness")
    q4.metric("High brightness",     iq.get("high_brightness_count", 0),
              help="> 220 mean brightness")

    extreme_aspect = iq.get("extreme_aspect_count", 0)
    if extreme_aspect:
        st.warning(f"{extreme_aspect} image(s) have extreme aspect ratio (> 4:1).")
    else:
        st.success("No extreme aspect ratio images (all within 4:1).")
else:
    st.info("Image quality stats not available in EDA summary.")

st.divider()

# ── Section 3: Detection density ─────────────────────────────────────────────
st.subheader("Detection Density")

det_info = eda.get("detection", {})
if det_info:
    d1, d2, d3 = st.columns(3)
    d1.metric("Total bbox annotations", det_info.get("total_bbox_annotations", "—"))
    d2.metric("Mean objects / image",   f"{det_info.get('mean_objects_per_image', 0):.2f}")
    d3.metric("Max objects in one image", det_info.get("max_objects_per_image", "—"))

    st.caption(
        "Average 6.81 objects per image reflects COCO's multi-object annotation style. "
        "YOLOv8n must detect all instances simultaneously — harder than single-object classification."
    )
else:
    st.info("Detection density stats not available in EDA summary.")
