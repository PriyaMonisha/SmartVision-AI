---
title: SmartVision AI
emoji: 👁️
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.37.0"
python_version: "3.11"
app_file: streamlit_app.py
pinned: false
---

# SmartVision AI — Intelligent Multi-Class Object Recognition System

> **2,500 COCO images · 4 CNN classifiers · YOLOv8 detection · Full MLOps stack**

![CI](https://github.com/PriyaMonisha/Smart-Vision-AI/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.14-0194E2?style=flat&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=flat&logo=huggingface&logoColor=black)

A production-grade computer vision platform that classifies objects into 25 COCO categories using 4 pretrained CNN architectures and detects multiple objects in a scene using YOLOv8. Covers the complete ML lifecycle: data streaming → preprocessing → transfer learning → detection → serving → monitoring → interactive dashboard.

**Engineering highlights:** Redis cache-aside with 24h TTL · KS-test double-gate drift detection · weighted 3-model ensemble · Prometheus metrics + Grafana dashboards · 57 automated tests · GitHub Actions CI · HuggingFace Spaces deployment

**Live Demo:** [huggingface.co/spaces/Moni38/smartvision-ai](https://huggingface.co/spaces/Moni38/smartvision-ai)

---

## What This Project Does

| Layer | What's Built |
|---|---|
| **Dataset Acquisition** | Streams COCO 2017 from HuggingFace — no 165 GB download. Checkpoint/resume support. 100 images/class × 25 classes = 2,500 total |
| **Preprocessing** | Crops bounding boxes → 224×224 classification dataset. YOLO format annotations. 70/15/15 train/val/test split |
| **Augmentation** | 10-transform pipeline: flips, rotation, colour jitter, RandomZoomOut, RandomErasing, ImageNet normalisation |
| **Transfer Learning** | 4 CNNs (VGG16, ResNet50, MobileNetV2, EfficientNetB0) with 2-phase training: frozen head → selective backbone fine-tuning |
| **Object Detection** | YOLOv8n fine-tuned on 25-class COCO subset for 50 epochs on T4 GPU. Per-class AP50, confusion matrix, training curves |
| **Model Ensemble** | POST /ensemble: weighted average of ResNet50 + EfficientNetB0 + MobileNetV2 softmax probabilities (weights ∝ test accuracy) |
| **Experiment Tracking** | MLflow with 5 logged runs, SQLite backend, JSON export summaries |
| **Inference API** | FastAPI serving /classify, /ensemble, /detect with Redis cache-aside and Prometheus metrics |
| **Drift Detection** | KS-test on confidence score distributions — double-gate (stat > 0.10 AND p < 0.05). 22 Prometheus Gauge series |
| **Dashboard** | Streamlit app with **7 pages** — loads pre-computed artifacts; classify/detect/webcam pages call FastAPI |
| **Monitoring** | Prometheus scraping + Grafana dashboards. Airflow DAG for drift-triggered retraining |
| **Tests** | 57 automated tests across health, classify, detect, drift, and Redis cache layers |

---

## Architecture

```
Google Colab T4 (Training)              Docker Compose (Serving)
──────────────────────────              ──────────────────────────────────────────────────
HuggingFace COCO Stream
        │
        ▼
Preprocess + Augment                    ┌─────────────────────────────────────────┐
        │                               │  Streamlit (port 8501)                  │
        ▼                               │  7 pages — calls FastAPI only           │
Train 4 CNNs + YOLOv8n                  └────────────────┬────────────────────────┘
        │                                                │  HTTP
        ▼                               ┌───────────────▼──────────────────────── ┐
HuggingFace Hub ──────weights──────────►│  FastAPI (port 8000)                    │
(model weights)                         │  /classify  /ensemble  /detect          │
                                        │  /health    /metrics   /drift/status    │
                                        └───┬─────────────┬───────────────────────┘
                                            │             │
                                    ┌───────▼───┐  ┌──────▼──────┐
                                    │  Redis 7  │  │ Prometheus  │
                                    │  cache    │  │ /metrics    │
                                    └───────────┘  └──────┬──────┘
                                                          │
                                                   ┌──────▼──────┐
                                                   │   Grafana   │
                                                   │  dashboards │
                                                   └─────────────┘
```

---

## Dashboard Pages

| Page | Requires FastAPI | Key Content |
|---|---|---|
| **Home** | Status badge only | Champion model stats, project overview, page navigation |
| **1 — Classify** | Yes | Single-model (ResNet50/MobileNet) + Ensemble tab with per-model breakdown |
| **2 — Detect** | Yes | YOLOv8n bounding boxes on EXIF-corrected image, 24-colour class palette |
| **3 — Model Comparison** | No | Accuracy bar, speed-accuracy scatter, YOLO per-class AP50, full metrics table |
| **4 — Drift Monitor** | Yes | Per-class KS drift gauge, alert sorting, manual refresh |
| **5 — EDA Insights** | No | Class distribution, image quality, bbox density, co-occurrence — pre-computed charts |
| **6 — About** | No | Dataset breakdown, model performance tabs, tech stack, engineering decisions |
| **7 — Webcam** | Yes | `st.camera_input()` → YOLOv8n detection, continuous mode toggle |

---

## ML Models

### Classification (Transfer Learning — 2-phase training)

| Model | Test Accuracy | Precision | F1 | CPU Latency | Size |
|---|---|---|---|---|---|
| **ResNet50** ← champion | **65.5%** | 66.6% | 65.5% | 115 ms | 94.5 MB |
| EfficientNetB0 | 58.9% | 59.3% | 58.3% | 51 ms | 16.4 MB |
| VGG16 | 59.5% | — | — | 229 ms | 527.8 MB |
| MobileNetV2 | 56.7% | 57.6% | — | 39 ms | 9.3 MB |

**Training approach:** Phase 1 — freeze backbone, train classification head (6–10 epochs). Phase 2 — unfreeze last backbone block, fine-tune with 10× lower learning rate (15–19 epochs).

> Accuracy ceiling reflects data volume: ~70 training images/class after 70% split on a 100-image dataset. Full-scale training (400+ images/class) projects ResNet50 to 70–75%.

### Ensemble (POST /ensemble)

Weighted average of ResNet50 + EfficientNetB0 + MobileNetV2 softmax probabilities. Weights are proportional to test accuracy: 36.2% / 32.5% / 31.3%. Returns per-model breakdown alongside the ensemble top-K.

### Object Detection (YOLOv8n)

| Metric | Value |
|---|---|
| mAP@0.5 | 14.7% |
| mAP@0.5:0.95 | 5.75% |
| Precision | 54.9% |
| Recall | 18.4% |
| CPU Latency (incl. NMS) | 154 ms |
| Epochs | 50 (T4 GPU) |

**Best classes:** cat 49.6% · pizza 34.2% · bed 32.5% · airplane 34.0% · bus 30.9%

---

## Tech Stack

```
Python 3.11          PyTorch 2.3.0         torchvision 0.18.0
Ultralytics 8.2.0    FastAPI 0.111.0        Redis 7
Streamlit 1.37.0     Plotly 5.22.0          MLflow 2.14.1
Prometheus client    Grafana latest         pytest 8.3.2
Docker Compose       GitHub Actions         HuggingFace Hub
```

---

## Project Structure

```
SmartVision-AI/
├── streamlit_app.py                    # Home page — health badge, champion stats
├── pages/
│   ├── 1_Classify.py                   # Single-model + ensemble tabs
│   ├── 2_Detect.py                     # YOLOv8n bbox drawing (EXIF-corrected)
│   ├── 3_Model_Comparison.py           # Accuracy, speed-accuracy, YOLO metrics
│   ├── 4_Drift_Monitor.py              # KS drift gauge, alert table
│   ├── 5_EDA_Insights.py               # Pre-computed EDA charts
│   ├── 6_About.py                      # Dataset, models, tech stack
│   └── 7_Webcam.py                     # st.camera_input → /detect
├── api/
│   ├── main.py                         # FastAPI lifespan — model loading, middleware
│   ├── schemas.py                      # Pydantic request/response models
│   ├── prometheus_metrics.py           # Counters, Gauges, Histograms
│   └── routes/
│       ├── classify.py                 # POST /classify (Redis cache-aside)
│       ├── ensemble.py                 # POST /ensemble (3-model weighted avg)
│       ├── detect.py                   # POST /detect (YOLOv8n + NMS)
│       ├── drift.py                    # GET /drift/status
│       ├── health.py                   # GET /health
│       └── metrics.py                  # GET /metrics (Prometheus text)
├── src/
│   ├── data/                           # preprocessor.py, augmentor.py, dataset.py, loader.py
│   ├── models/                         # base_classifier.py, model_factory.py
│   ├── inference/                      # model_loader.py, redis_cache.py
│   ├── monitoring/                     # drift_detector.py (KS double-gate)
│   └── utils/                          # helpers.py (NumpyEncoder)
├── streamlit_app/
│   ├── api_client.py                   # HTTP client — demo_banner(), is_hf_spaces()
│   └── plotting.py                     # accuracy_bar, speed_accuracy_scatter, drift_gauge
├── notebooks/                          # 8 training/analysis notebooks (.py + .ipynb)
├── tests/                              # 57 tests across 5 files
├── artifacts/                          # Pre-computed metrics JSONs, PNGs, .npy baselines
├── monitoring/
│   ├── prometheus.yml                  # Scrape config (fastapi:8000/metrics, 15s)
│   └── grafana/                        # Provisioned datasource + 4-panel dashboard
├── dags/
│   └── retrain_drift_dag.py            # Airflow DAG — drift-triggered retraining
├── config.py                           # All constants via Pydantic Settings
├── docker-compose.yml                  # 5-service stack (redis·fastapi·streamlit·prometheus·grafana)
├── docker-compose.airflow.yml          # Optional Airflow overlay (4 services)
├── Dockerfile                          # python:3.11-slim, non-root appuser
├── spaces.yaml                         # HuggingFace Spaces config
├── packages.txt                        # HF system deps (libgl1, libglib2.0-0)
├── requirements.txt                    # Full dependencies
├── requirements-hf.txt                 # CPU-only PyTorch for HF Spaces
└── .github/workflows/ci.yml            # Lint (ruff) → 57 tests → coverage artifact
```

---

## Quick Start

### Verify the code — no model weights needed

```bash
git clone https://github.com/PriyaMonisha/Smart-Vision-AI.git
cd Smart-Vision-AI
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
pytest tests/ -v               # 57 tests — uses fixtures, no weights required
```

---

### Option A — Streamlit dashboard only (2 minutes, no Docker)

Pre-computed artifacts are committed — Model Comparison, EDA, and About pages work instantly.

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Dashboard opens at **http://localhost:8501** · Classify/Detect pages require Option B.

---

### Option B — Full MLOps stack via Docker

```bash
# Copy and fill in HuggingFace credentials
copy .env.example .env
# Set HF_TOKEN and HF_REPO_ID in .env

docker compose up --build -d
```

| Service | URL | Purpose |
|---|---|---|
| Streamlit | http://localhost:8501 | Interactive dashboard |
| FastAPI | http://localhost:8000 | Model inference API |
| Prometheus | http://localhost:9090 | Metrics collector |
| Grafana | http://localhost:3000 | Live dashboards (admin/admin) |

```bash
docker compose logs -f fastapi     # stream API logs
docker compose ps                  # check service health
docker compose down                # stop all services
```

---

### Option C — FastAPI only (local development)

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

---

### Reproduce Training

Training runs on Google Colab T4. Convert notebooks with jupytext, then open in Colab:

```bash
jupytext --to notebook notebooks/04_train_classifier.py
jupytext --to notebook notebooks/05_yolo_training.py
```

Set `FAST_MODE = False` at the top of each notebook for full training. Weights upload automatically to HuggingFace Hub after training.

---

## API Endpoints

```bash
# Single-model classification
curl -X POST http://localhost:8000/classify \
  -F "file=@image.jpg" -F "model_name=resnet50" -F "top_k=5"

# Ensemble classification (3 CNNs)
curl -X POST http://localhost:8000/ensemble \
  -F "file=@image.jpg" -F "top_k=5"

# Object detection
curl -X POST http://localhost:8000/detect \
  -F "file=@image.jpg"
```

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service health + loaded models list |
| GET | `/metrics` | Prometheus metrics (text/plain) |
| POST | `/classify` | Top-K classification (ResNet50 or MobileNet) |
| POST | `/ensemble` | Weighted 3-model ensemble classification |
| POST | `/detect` | YOLOv8n object detection with bounding boxes |
| GET | `/drift/status` | Per-class KS drift status vs baseline |

API docs at **http://localhost:8000/docs**

---

## Tests

```bash
pytest tests/ -v                           # 57 tests
pytest tests/ --cov=api --cov=src          # with coverage report
```

| File | Tests | Coverage |
|---|---|---|
| `test_api_health.py` | 8 | Health endpoint — ready/loading/503 states |
| `test_api_classify.py` | 14 | Classify route — valid input, bad model name, Redis cache hit/miss |
| `test_api_detect.py` | 11 | Detect route — valid image, bad input, bbox response format |
| `test_drift_detector.py` | 14 | KS test, double-gate logic, baseline loading, deque fallback |
| `test_redis_cache.py` | 10 | Cache set/get/TTL, fakeredis injection, graceful Redis failure |

CI runs on every push via GitHub Actions (`.github/workflows/ci.yml`): ruff lint → ruff format check → pytest → coverage artifact.

---

## Production Readiness

| Category | What's Implemented |
|---|---|
| **Security** | Redis password support · no secrets in VCS · `.env.example` documents all required vars |
| **Performance** | Models preloaded at startup via FastAPI lifespan · Redis cache-aside (24h classify, 1h detect) · async request handling |
| **Reliability** | Health checks on all 5 Docker services · graceful Redis degradation (never raises on cache miss) · 57 automated tests |
| **MLOps** | MLflow experiment tracking (5 runs) · model weights on HuggingFace Hub · Airflow drift-triggered retraining DAG |
| **Observability** | Prometheus counters (requests, cache hits/misses, drift alerts, HTTP errors) · Gauge (models loaded) · Grafana 4-panel dashboard · auto-provisioned datasource |
| **Reproducibility** | Fixed `RANDOM_STATE=42` across all splits and models · COCO_ID_TO_CLASS_IDX mapping validated at startup |

---

## Key Design Decisions

- **Streamlit never loads models** — all inference goes through FastAPI endpoints. Pages are display-only; this enforces a clean separation between serving and UI layers.
- **Redis is optional** — the API degrades gracefully if Redis is unavailable. Cache misses fall through to inference; a `RedisError` never surfaces to the user.
- **KS drift uses confidence scores, not embeddings** — 1-dimensional, interpretable, and computationally free at inference time. Double-gate (stat > 0.10 AND p < 0.05) eliminates false positives with small baseline samples (n=30).
- **Weighted ensemble over majority vote** — softmax average preserves confidence information that majority vote discards. Weights proportional to test accuracy give stronger models more say.
- **lifespan context manager over @app.on_event** — uses the modern FastAPI pattern; model loading runs in a thread pool via `run_in_executor` so the event loop stays responsive during the 1–3 minute startup window.
- **EXIF orientation correction on both sides** — `ImageOps.exif_transpose` is applied to the PIL image before YOLO inference AND before bbox drawing, so bounding box coordinates always align with the visually correct orientation.
- **Grafana provisioned from config** — datasource and dashboard auto-load on container start with zero manual setup. `access: proxy` is required (not `direct`) because Docker internal hostnames are not resolvable from the browser.
- **Training in Colab, weights on HF Hub** — avoids Git LFS entirely. Weights are downloaded at container startup via `hf_hub_download()`.

---

## Dataset

| Item | Value |
|---|---|
| Source | COCO 2017 via `detection-datasets/coco` on HuggingFace |
| Acquisition | Streaming — no full 165 GB download |
| Total images | 2,500 (100/class × 25 classes) |
| Train / Val / Test | 1,750 / 375 / 375 (70/15/15) |
| Classification input | 224×224 cropped objects, ImageNet normalised |
| Detection input | 640×640 full scenes, YOLO format annotations |

**25 Classes:**

| Category | Classes |
|---|---|
| Vehicles (6) | car · truck · bus · motorcycle · bicycle · airplane |
| Person (1) | person |
| Outdoor (3) | traffic light · stop sign · bench |
| Animals (6) | dog · cat · horse · bird · cow · elephant |
| Kitchen & Food (5) | bottle · cup · bowl · pizza · cake |
| Furniture (4) | chair · couch · bed · potted plant |

---

## Author

**Priya Monisha** · [GitHub](https://github.com/PriyaMonisha) · [HuggingFace](https://huggingface.co/Moni38)
