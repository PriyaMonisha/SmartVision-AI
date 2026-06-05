# filename: pages/1_Classify.py
# purpose:  Upload an image → POST /classify → top-K confidence bar chart.
#           No model calls. Result-first layout (Rule 12). Single-read file cursor (plan C2).

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config
from streamlit_app import api_client

st.set_page_config(page_title="Classify — SmartVision AI", layout="wide")
st.title("Image Classification")
st.caption("ResNet50 (65.5%) or MobileNet (56.7%) via FastAPI /classify.")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", ["resnet50", "mobilenet"], index=0)
    top_k      = st.slider("Top-K predictions", min_value=3, max_value=10, value=5)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    help=f"Max {config.MAX_IMAGE_SIZE_MB} MB, max {config.MAX_IMAGE_RESOLUTION}px on longest side.",
)

if uploaded_file is None:
    st.info("Upload an image to run classification.")
    st.stop()

# Read ONCE into bytes — avoids cursor-advance bug if opened twice (plan C2)
img_bytes = uploaded_file.read()

# ── Validation ────────────────────────────────────────────────────────────────
if len(img_bytes) > config.MAX_IMAGE_SIZE_MB * 1024 ** 2:
    st.error(f"Image too large (max {config.MAX_IMAGE_SIZE_MB} MB). Please resize and retry.")
    st.stop()

image = Image.open(io.BytesIO(img_bytes))
w, h  = image.size
if max(w, h) > config.MAX_IMAGE_RESOLUTION:
    st.error(
        f"Image resolution too large ({w}×{h} px, max {config.MAX_IMAGE_RESOLUTION}px). "
        "Please resize and retry."
    )
    st.stop()

# ── Inference ─────────────────────────────────────────────────────────────────
with st.spinner(f"Running {model_name.upper()} inference…"):
    try:
        result = api_client.classify(img_bytes, uploaded_file.name, model_name, top_k)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

# ── Result-first layout (Rule 12) ─────────────────────────────────────────────
predictions   = result.get("predictions", [])
inf_time      = result.get("inference_time_ms", 0.0)
cached        = result.get("cached", False)

col_img, col_chart = st.columns([1, 2])

with col_img:
    st.image(image, caption=uploaded_file.name, use_container_width=True)
    badge_col, time_col = st.columns(2)
    if cached:
        badge_col.success("Cache hit")
    else:
        badge_col.info("Fresh inference")
    time_col.metric("Inference time", f"{inf_time:.1f} ms")

with col_chart:
    if not predictions:
        st.warning("No predictions returned.")
    else:
        classes = [p["class_name"] for p in predictions]
        confs   = [p["confidence"] for p in predictions]

        fig = go.Figure(go.Bar(
            x=confs,
            y=classes,
            orientation="h",
            marker_color="#636EFA",
            text=[f"{c:.1%}" for c in confs],
            textposition="outside",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig.update_layout(
            title=f"Top-{top_k} Predictions ({model_name.upper()})",
            xaxis=dict(title="Confidence", tickformat=".0%", range=[0, 1.1]),
            yaxis=dict(title="", autorange="reversed"),
            height=max(250, top_k * 45),
            margin=dict(l=10, r=70, t=50, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Technical details expander (Rule 12) ──────────────────────────────────────
with st.expander("Technical details"):
    st.json(result)
    st.caption(
        f"Model: {result.get('model_name', model_name)} | "
        f"Inference: {inf_time:.1f} ms | "
        f"Cached: {cached}"
    )
