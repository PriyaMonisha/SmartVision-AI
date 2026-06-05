# filename: pages/3_Model_Comparison.py
# purpose:  Static comparison dashboard from model_metrics.json + yolo_metrics.json.
#           No API calls — loads local artifacts. Cache invalidates on file change (plan C4).

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
from streamlit_app.plotting import accuracy_bar, speed_accuracy_scatter

COMPARISON_PATH   = config.ARTIFACTS_DIR / "comparison" / "model_metrics.json"
YOLO_METRICS_PATH = config.ARTIFACTS_DIR / "detection" / "yolo_metrics.json"

st.set_page_config(page_title="Model Comparison — SmartVision AI", layout="wide")
st.title("Model Comparison")
st.caption("Loaded from local artifacts — no FastAPI required.")

# ── File-mtime cache invalidation (plan C4) ───────────────────────────────────
@st.cache_data
def _load_json(path_str: str, mtime: float) -> dict:
    # mtime as cache key: invalidates when file changes (2s resolution on FAT32, not a concern here)
    return json.loads(Path(path_str).read_text())


def load_metrics() -> dict:
    return _load_json(str(COMPARISON_PATH), COMPARISON_PATH.stat().st_mtime)


def load_yolo_metrics() -> dict:
    if not YOLO_METRICS_PATH.exists():
        return {}
    return _load_json(str(YOLO_METRICS_PATH), YOLO_METRICS_PATH.stat().st_mtime)


# ── Load artifacts ────────────────────────────────────────────────────────────
if not COMPARISON_PATH.exists():
    st.error("model_metrics.json not found. Run notebooks/06_model_comparison.py first.")
    st.stop()

metrics      = load_metrics()
yolo_metrics = load_yolo_metrics()

champion     = metrics.get("champion_classifier", "resnet50")
champion_acc = metrics.get("champion_test_accuracy", 0.0)
models_data  = metrics.get("models", {})

# ── Champion callout ──────────────────────────────────────────────────────────
st.metric(
    label=f"Champion Classifier: {champion.upper()}",
    value=f"{champion_acc:.1%} test accuracy",
)
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Accuracy", "Speed & Accuracy", "Detection", "Full Table"])

# ── Tab 1: Accuracy bar ───────────────────────────────────────────────────────
with tab1:
    st.plotly_chart(accuracy_bar(metrics), use_container_width=True)
    st.caption(
        f"Red bar = champion ({champion.upper()}). "
        "VGG16 metrics are from Round 1 (69 img/class); others from Round 2 (200 img/class)."
    )

# ── Tab 2: Speed vs accuracy scatter ─────────────────────────────────────────
with tab2:
    st.plotly_chart(speed_accuracy_scatter(metrics), use_container_width=True)
    st.caption(
        "CPU latency = architecture-only benchmark (pretrained=False). "
        "Correct topology (same Dropout + head as training). "
        "Latency is a function of architecture, not learned weights."
    )

# ── Tab 3: Detection metrics ──────────────────────────────────────────────────
with tab3:
    det = metrics.get("detection", {}).get("yolov8n", {})
    if det:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("mAP50",      f"{det.get('map50', 0):.1%}")
        d2.metric("mAP50-95",   f"{det.get('map50_95', 0):.1%}")
        d3.metric("Precision",  f"{det.get('precision', 0):.1%}")
        d4.metric("Recall",     f"{det.get('recall', 0):.1%}")
        st.caption(
            f"YOLOv8n trained for {det.get('epochs_trained', 50)} epochs on "
            "COCO 2017 subset (22 classes, ~140 img/class). "
            "CPU inference includes NMS post-processing."
        )

    if yolo_metrics:
        per_class = yolo_metrics.get("per_class", {})
        if per_class:
            st.subheader("Per-Class AP50")
            pc_data = [
                {"Class": cls, "AP50": v.get("ap50", 0)}
                for cls, v in sorted(per_class.items(), key=lambda x: x[1].get("ap50", 0), reverse=True)
            ]
            pc_df = pd.DataFrame(pc_data)
            fig_pc = go.Figure(go.Bar(
                x=pc_df["AP50"],
                y=pc_df["Class"],
                orientation="h",
                marker_color="#00CC96",
                text=[f"{v:.1%}" for v in pc_df["AP50"]],
                textposition="outside",
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            fig_pc.update_layout(
                xaxis=dict(title="AP50", tickformat=".0%", range=[0, max(pc_df["AP50"]) * 1.2]),
                yaxis=dict(title=""),
                height=max(300, len(pc_data) * 22),
                margin=dict(l=20, r=60, t=20, b=30),
            )
            st.plotly_chart(fig_pc, use_container_width=True)
    else:
        st.info("yolo_metrics.json not found — per-class AP50 chart unavailable.")

# ── Tab 4: Full table ─────────────────────────────────────────────────────────
with tab4:
    rows = []
    for name, d in models_data.items():
        rows.append({
            "Model":          name.upper(),
            "Test Acc":       f"{d['test_accuracy']:.1%}" if d.get("test_accuracy") else "—",
            "Val Acc":        f"{d['val_accuracy']:.1%}"  if d.get("val_accuracy")  else "—",
            "Precision":      f"{d['test_precision']:.4f}" if d.get("test_precision") else "—",
            "Recall":         f"{d['test_recall']:.4f}"    if d.get("test_recall")    else "—",
            "F1":             f"{d['test_f1']:.4f}"        if d.get("test_f1")        else "—",
            "Size (MB)":      f"{d['model_size_mb']:.1f}"  if d.get("model_size_mb")  else "—",
            "CPU lat. (ms)":  f"{d['cpu_inference_ms']:.1f}" if d.get("cpu_inference_ms") else "—",
            "Epochs":         d.get("epochs_trained", "—"),
        })

    # Add YOLO row
    det = metrics.get("detection", {}).get("yolov8n", {})
    if det:
        rows.append({
            "Model":         "YOLOV8N",
            "Test Acc":      f"mAP50={det.get('map50', 0):.1%}",
            "Val Acc":       "—",
            "Precision":     f"{det.get('precision', 0):.4f}",
            "Recall":        f"{det.get('recall', 0):.4f}",
            "F1":            "—",
            "Size (MB)":     "~6",
            "CPU lat. (ms)": f"{det.get('cpu_inference_ms', 0):.1f} (incl NMS)",
            "Epochs":        det.get("epochs_trained", "—"),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Latency = architecture-only CPU benchmark. "
        "VGG16: Round 1 (69 img/class). Others: Round 2 (200 img/class)."
    )
