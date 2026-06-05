# filename: pages/1_Classify.py
# purpose:  Upload an image → POST /classify or /ensemble → top-K confidence bar chart.
#           Single tab = single model. Ensemble tab = 3-model weighted average.
#           No model calls. Result-first layout (Rule 12). Single-read file cursor (plan C2).

from __future__ import annotations

import io
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
st.caption("Single-model or weighted ensemble via FastAPI.")
api_client.demo_banner()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K predictions", min_value=3, max_value=10, value=5)
    st.divider()
    st.markdown("""
    **Single model:** choose ResNet50 or MobileNet.

    **Ensemble:** weighted average of ResNet50 + EfficientNetB0 + MobileNetV2
    (weights proportional to test accuracy).
    """)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    help=f"Max {config.MAX_IMAGE_SIZE_MB} MB.",
)

if uploaded_file is None:
    st.info("Upload an image to run classification.")
    st.stop()

img_bytes = uploaded_file.read()

if len(img_bytes) > config.MAX_IMAGE_SIZE_MB * 1024 ** 2:
    st.error(f"Image too large (max {config.MAX_IMAGE_SIZE_MB} MB). Please resize and retry.")
    st.stop()

image = Image.open(io.BytesIO(img_bytes))
w, h  = image.size
if max(w, h) > config.MAX_IMAGE_RESOLUTION:
    st.error(f"Image too large ({w}×{h} px, max {config.MAX_IMAGE_RESOLUTION}px).")
    st.stop()

# ── Tabs: Single Model | Ensemble ─────────────────────────────────────────────
tab_single, tab_ensemble = st.tabs(["Single Model", "Ensemble (3 CNNs)"])

# ── TAB 1: Single model ───────────────────────────────────────────────────────
with tab_single:
    model_name = st.selectbox("Model", ["resnet50", "mobilenet"], index=0)

    with st.spinner(f"Running {model_name.upper()} inference…"):
        try:
            result = api_client.classify(img_bytes, uploaded_file.name, model_name, top_k)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

    predictions = result.get("predictions", [])
    inf_time    = result.get("inference_time_ms", 0.0)
    cached      = result.get("cached", False)

    col_img, col_chart = st.columns([1, 2])

    with col_img:
        st.image(image, caption=uploaded_file.name, use_container_width=True)
        b_col, t_col = st.columns(2)
        b_col.success("Cache hit") if cached else b_col.info("Fresh inference")
        t_col.metric("Inference", f"{inf_time:.1f} ms")

    with col_chart:
        if predictions:
            fig = go.Figure(go.Bar(
                x=[p["confidence"] for p in predictions],
                y=[p["class_name"]  for p in predictions],
                orientation="h",
                marker_color="#636EFA",
                text=[f"{p['confidence']:.1%}" for p in predictions],
                textposition="outside",
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            fig.update_layout(
                title=f"Top-{top_k} ({model_name.upper()})",
                xaxis=dict(title="Confidence", tickformat=".0%", range=[0, 1.1]),
                yaxis=dict(autorange="reversed"),
                height=max(250, top_k * 45),
                margin=dict(l=10, r=70, t=50, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Technical details"):
        st.json(result)

# ── TAB 2: Ensemble ───────────────────────────────────────────────────────────
with tab_ensemble:
    st.caption("ResNet50 (36.2%) + EfficientNetB0 (32.5%) + MobileNetV2 (31.3%) — weights by test accuracy.")

    with st.spinner("Running ensemble inference (3 models)…"):
        try:
            result = api_client.ensemble(img_bytes, uploaded_file.name, top_k)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

    ens_preds  = result.get("ensemble_predictions", [])
    per_model  = result.get("per_model", {})
    inf_time   = result.get("inference_time_ms", 0.0)
    cached     = result.get("cached", False)
    models_used = result.get("models_used", [])

    # ── Top metrics ────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    if ens_preds:
        col1.metric("Top prediction", ens_preds[0]["class_name"])
        col1.metric("Ensemble confidence", f"{ens_preds[0]['confidence']:.1%}")
    col2.metric("Inference time", f"{inf_time:.1f} ms")
    col2.metric("Models used", len(models_used))
    col3.success("Cache hit") if cached else col3.info("Fresh inference")

    col_img, col_chart = st.columns([1, 2])

    with col_img:
        st.image(image, caption=uploaded_file.name, use_container_width=True)

    with col_chart:
        if ens_preds:
            fig = go.Figure(go.Bar(
                x=[p["confidence"] for p in ens_preds],
                y=[p["class_name"]  for p in ens_preds],
                orientation="h",
                marker_color="#EF553B",
                text=[f"{p['confidence']:.1%}" for p in ens_preds],
                textposition="outside",
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            fig.update_layout(
                title=f"Ensemble Top-{top_k} (weighted average)",
                xaxis=dict(title="Confidence", tickformat=".0%", range=[0, 1.1]),
                yaxis=dict(autorange="reversed"),
                height=max(250, top_k * 45),
                margin=dict(l=10, r=70, t=50, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Per-model breakdown ────────────────────────────────────────────────────
    if per_model:
        st.subheader("Per-model breakdown")
        cols = st.columns(len(per_model))
        colors = {"resnet50": "#636EFA", "efficientnet": "#00CC96", "mobilenet": "#AB63FA"}

        for col, (mname, preds) in zip(cols, per_model.items()):
            with col:
                top1 = preds[0]["class_name"] if preds else "—"
                conf = preds[0]["confidence"]  if preds else 0.0
                st.metric(mname.upper(), top1, f"{conf:.1%}")
                fig_m = go.Figure(go.Bar(
                    x=[p["confidence"] for p in preds],
                    y=[p["class_name"]  for p in preds],
                    orientation="h",
                    marker_color=colors.get(mname, "#636EFA"),
                    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
                ))
                fig_m.update_layout(
                    height=200,
                    xaxis=dict(tickformat=".0%", range=[0, 1.0]),
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=5, r=5, t=5, b=5),
                    showlegend=False,
                )
                st.plotly_chart(fig_m, use_container_width=True)

    with st.expander("Technical details"):
        st.json(result)
