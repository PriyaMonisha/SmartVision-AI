# filename: pages/6_About.py
# purpose:  Project documentation — dataset, architecture, model performance,
#           tech stack. Static page, no API calls required.

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="About — SmartVision AI", layout="wide")
st.title("About SmartVision AI")
st.caption("Project documentation, dataset details, architecture, and performance metrics.")

# ── Project Overview ──────────────────────────────────────────────────────────
st.header("Project Overview")
st.markdown("""
SmartVision AI is a production-grade computer vision platform that combines
**image classification** and **object detection** into a fully monitored, containerised serving stack.

**Two core capabilities:**
- **Classify** — 4 pretrained CNNs (VGG16, ResNet50, MobileNetV2, EfficientNetB0) identify a single
  dominant object from 25 COCO categories with top-K confidence scores.
- **Detect** — YOLOv8n localises multiple objects in a scene, drawing bounding boxes with class
  labels and confidence scores.

**Production stack:** FastAPI serves inference → Redis caches results → Prometheus scrapes metrics
→ Grafana dashboards → Streamlit provides the user interface. Streamlit never loads models directly.
""")

# ── Architecture ──────────────────────────────────────────────────────────────
st.header("Architecture")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Training Pipeline (Google Colab T4)")
    st.markdown("""
    1. Stream COCO 2017 from HuggingFace (no 165 GB download)
    2. Filter 25 classes, collect 100 images/class → 2,500 total
    3. Crop bounding boxes → 224×224 classification dataset
    4. Prepare YOLO format annotations for detection
    5. Train 4 CNNs with transfer learning + fine-tuning
    6. Train YOLOv8n for 50 epochs
    7. Upload weights to HuggingFace Hub
    8. Commit metrics JSON to git
    """)

with col2:
    st.subheader("Serving Stack (Docker Compose)")
    st.markdown("""
    - **FastAPI** — loads weights from HF Hub at startup, lifespan context manager
    - **Redis** — cache-aside with 24h TTL for classify, 1h for detect
    - **Drift Detector** — KS-test on confidence score distributions (double-gate: stat > 0.10 AND p < 0.05)
    - **Prometheus** — scrapes `/metrics` every 15s, fires alerts on drift + high latency
    - **Grafana** — 4-panel dashboard provisioned automatically from config
    - **Airflow** — optional retraining DAG triggered on drift alert
    """)

# ── Dataset ───────────────────────────────────────────────────────────────────
st.header("Dataset")

st.markdown("""
**COCO 2017** (Common Objects in Context) — industry benchmark for object detection and classification.
Streamed from `detection-datasets/coco` on HuggingFace. No full 165 GB download required.
""")

classes_data = {
    "Category": ["Vehicles", "Person", "Outdoor", "Animals", "Kitchen & Food", "Furniture"],
    "Classes": [
        "car, truck, bus, motorcycle, bicycle, airplane",
        "person",
        "traffic light, stop sign, bench",
        "dog, cat, horse, bird, cow, elephant",
        "bottle, cup, bowl, pizza, cake",
        "chair, couch, bed, potted plant",
    ],
    "Count": [6, 1, 3, 6, 5, 4],
}

st.dataframe(
    pd.DataFrame(classes_data),
    use_container_width=True,
    hide_index=True,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Images", "2,500")
col2.metric("Classes", "25")
col3.metric("Train / Val / Test", "70 / 15 / 15 %")
col4.metric("Images per Class", "100")

# ── Model Performance ─────────────────────────────────────────────────────────
st.header("Model Performance")

tab1, tab2 = st.tabs(["Classification", "Detection"])

with tab1:
    clf_data = {
        "Model": ["ResNet50", "EfficientNetB0", "VGG16", "MobileNetV2"],
        "Test Accuracy": ["65.5%", "58.9%", "59.5%", "56.7%"],
        "Precision": ["66.6%", "59.3%", "—", "57.6%"],
        "F1 Score": ["65.5%", "58.3%", "—", "—"],
        "CPU Latency": ["115 ms", "51 ms", "229 ms", "39 ms"],
        "Model Size": ["94.5 MB", "16.4 MB", "527.8 MB", "9.3 MB"],
        "Champion": ["✅", "", "", ""],
    }
    st.dataframe(pd.DataFrame(clf_data), use_container_width=True, hide_index=True)
    st.info(
        "Accuracy ceiling reflects data volume: ~70 training images/class after 70% split. "
        "ResNet50's 2048-dim feature space extracts the strongest signal at this scale. "
        "Full-scale training (400+ images/class) projects to 70–75% accuracy.",
        icon="ℹ️",
    )

with tab2:
    det_data = {
        "Metric": ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "CPU Latency", "Epochs"],
        "YOLOv8n": ["14.7%", "5.75%", "54.9%", "18.4%", "154 ms (incl. NMS)", "50"],
    }
    st.dataframe(pd.DataFrame(det_data), use_container_width=True, hide_index=True)

    st.markdown("**Best performing classes:**")
    best_cls = {
        "Class": ["cat", "pizza", "bed", "airplane", "bus"],
        "AP@0.5": ["49.6%", "34.2%", "32.5%", "34.0%", "30.9%"],
        "Precision": ["84.8%", "81.3%", "83.3%", "75.0%", "61.4%"],
    }
    st.dataframe(pd.DataFrame(best_cls), use_container_width=True, hide_index=True)
    st.info(
        "mAP reflects training scale: 3,080 images across 22 classes (~140 images/class). "
        "The 75% mAP benchmark in the brief assumes COCO-scale data (118K images). "
        "YOLOv8n is the nano model — optimised for speed over accuracy.",
        icon="ℹ️",
    )

# ── Tech Stack ────────────────────────────────────────────────────────────────
st.header("Tech Stack")

stack_data = {
    "Layer": [
        "Language", "Deep Learning", "Object Detection", "Dataset",
        "API", "Cache", "Monitoring", "Experiment Tracking",
        "UI", "Testing", "CI/CD", "Containers", "Model Storage",
    ],
    "Technology": [
        "Python 3.11", "PyTorch + torchvision", "Ultralytics YOLOv8",
        "HuggingFace COCO (streaming)", "FastAPI + uvicorn", "Redis 7",
        "Prometheus + Grafana", "MLflow", "Streamlit + Plotly",
        "pytest + httpx", "GitHub Actions", "Docker Compose", "HuggingFace Hub",
    ],
    "Version": [
        "3.11", "2.3.0 / 0.18.0", "8.2.0", "2.20.0",
        "0.111.0 / 0.30.1", "7", "latest", "2.14.1",
        "1.37.0 / 5.22.0", "8.3.2 / 0.27.0", "—", "—", "—",
    ],
}
st.dataframe(pd.DataFrame(stack_data), use_container_width=True, hide_index=True)

# ── Beyond Requirements ───────────────────────────────────────────────────────
st.header("Production Additions")
st.markdown("""
This project goes beyond a standard capstone to demonstrate a full MLOps pipeline:

| Feature | What It Shows |
|---|---|
| FastAPI + Redis caching | Production API design, latency optimisation |
| Prometheus + Grafana | Observability, SRE discipline |
| KS-test drift detection | MLOps — knowing when to retrain |
| Apache Airflow DAG | Data pipeline orchestration |
| Docker Compose (6 services) | Container orchestration |
| 57 pytest tests, ~95% API coverage | Software engineering discipline |
| GitHub Actions CI | Automated lint + test on every push |
| MLflow experiment tracking | Reproducible ML experiments |
""")

# ── Key Engineering Decisions ─────────────────────────────────────────────────
with st.expander("Key Engineering Decisions"):
    st.markdown("""
    - **Streamlit never loads models** — all inference goes through FastAPI endpoints.
      Pages are display-only; models live in the API layer.
    - **Redis is optional** — the API degrades gracefully if Redis is unavailable.
      Cache misses fall through to inference, never raise.
    - **KS drift uses confidence scores** — not raw embeddings. 1-dimensional,
      interpretable, and computationally cheap at inference time.
    - **lifespan context manager** — FastAPI startup uses the modern lifespan API
      (not deprecated `@app.on_event`).
    - **COCO IDs are not sequential** — `COCO_ID_TO_CLASS_IDX` mapping applied
      everywhere to avoid label misalignment.
    - **EXIF orientation correction** — `ImageOps.exif_transpose` applied before
      YOLO inference so bounding boxes align with visually correct orientation.
    - **Grafana provisioned from config** — zero manual setup; datasource and
      dashboard load automatically on container start.
    """)
