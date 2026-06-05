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
suggested_hardware: cpu-basic
tags:
  - computer-vision
  - image-classification
  - object-detection
  - drift-monitoring
  - mlops
---

# SmartVision AI — Intelligent Multi-Class Object Recognition System

A production-grade computer vision platform combining 4 CNN classifiers and YOLOv8 object detection,
served through a full MLOps stack: FastAPI + Redis + Prometheus + Grafana + Streamlit.

**Live Demo:** [huggingface.co/spaces/Moni38/smartvision-ai](https://huggingface.co/spaces/Moni38/smartvision-ai)

---

## What It Does

| Capability | Details |
|---|---|
| Image Classification | 4 CNNs (VGG16, ResNet50, MobileNetV2, EfficientNetB0) on 25 COCO classes |
| Object Detection | YOLOv8n fine-tuned on 25-class COCO subset with bounding boxes |
| Drift Monitoring | KS-test on confidence score distributions, real-time Prometheus alerts |
| Model Comparison | Accuracy, latency, model size benchmarks with MLflow tracking |
| Production Serving | FastAPI + Redis cache-aside + Prometheus metrics + Grafana dashboards |

---

## Architecture

```
Google Colab (Training)          Docker Compose (Serving)
─────────────────────            ──────────────────────────────────────────
COCO HF Stream                   Streamlit ──► FastAPI ──► ResNet50 / YOLO
    │                                              │
    ▼                                              ├──► Redis (24h cache)
Preprocess + Augment                               ├──► Drift Detector (KS)
    │                                              └──► Prometheus /metrics
    ▼                                                        │
Train 4 CNNs + YOLOv8n                            Grafana Dashboards
    │
    ▼
HuggingFace Hub (weights)
```

---

## Model Performance

### Classification (ResNet50 is champion)

| Model | Test Accuracy | Precision | F1 | CPU Latency | Size |
|---|---|---|---|---|---|
| ResNet50 | **65.5%** | 66.6% | 65.5% | 115 ms | 94.5 MB |
| EfficientNetB0 | 58.9% | 59.3% | 58.3% | 51 ms | 16.4 MB |
| VGG16 | 59.5% | — | — | 229 ms | 527.8 MB |
| MobileNetV2 | 56.7% | 57.6% | — | 39 ms | 9.3 MB |

> Accuracy ceiling at 100 images/class (70% split = ~70 training images/class) is a known data volume constraint.
> Full-scale training (400+ images/class) is expected to reach 70–75% for ResNet50.

### Detection (YOLOv8n)

| Metric | Value |
|---|---|
| mAP@0.5 | 14.7% |
| mAP@0.5:0.95 | 5.75% |
| Precision | 54.9% |
| Recall | 18.4% |
| Best class | cat (49.6% AP), pizza (34.2%), bed (32.5%) |
| Training | 50 epochs on T4, 3,080 images |

---

## 25 Object Classes

| Category | Classes |
|---|---|
| Vehicles (6) | car, truck, bus, motorcycle, bicycle, airplane |
| Person (1) | person |
| Outdoor (3) | traffic light, stop sign, bench |
| Animals (6) | dog, cat, horse, bird, cow, elephant |
| Kitchen & Food (5) | bottle, cup, bowl, pizza, cake |
| Furniture (4) | chair, couch, bed, potted plant |

---

## Quick Start

### Option A — HuggingFace Spaces (no setup)

Visit the live demo link at the top of this README.

### Option B — Local (Streamlit only, no Docker)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The Streamlit app works standalone for Model Comparison and EDA pages.
Classification and Detection pages require the FastAPI backend (Option C).

### Option C — Full Stack (Docker Compose)

```bash
# 1. Copy and fill in your HuggingFace token
cp .env.example .env
# Edit .env: set HF_TOKEN and HF_REPO_ID

# 2. Start all services
docker compose up --build

# Services:
#   Streamlit  → http://localhost:8501
#   FastAPI    → http://localhost:8000
#   Prometheus → http://localhost:9090
#   Grafana    → http://localhost:3000  (admin/admin)
#   Redis      → localhost:6379
```

### Option D — FastAPI only (local development)

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Project Structure

```
smartvision-ai/
├── api/                        # FastAPI app
│   ├── main.py                 # lifespan startup, model loading
│   ├── routes/                 # classify, detect, health, metrics, drift
│   └── schemas.py              # Pydantic request/response models
├── src/
│   ├── data/                   # preprocessor, augmentor, dataset, loader
│   ├── models/                 # base_classifier, model_factory (4 CNNs)
│   ├── inference/              # model_loader, redis_cache
│   └── monitoring/             # drift_detector (KS-test)
├── streamlit_app/              # api_client, plotting utilities
├── pages/                      # 5 Streamlit pages
├── notebooks/                  # 8 training/analysis notebooks (.py + .ipynb)
├── monitoring/                 # Prometheus rules + Grafana provisioning
├── dags/                       # Airflow retrain DAG
├── tests/                      # 57 pytest tests (~95% API coverage)
├── artifacts/                  # model metrics, EDA figures, drift baselines
├── docker-compose.yml          # 5-service production stack
├── docker-compose.airflow.yml  # optional Airflow scheduler
├── Dockerfile                  # python:3.11-slim, non-root appuser
├── spaces.yaml                 # HuggingFace Spaces config
└── config.py                   # Pydantic Settings, all constants
```

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.11 |
| Deep Learning | PyTorch + torchvision | 2.3.0 / 0.18.0 |
| Object Detection | Ultralytics YOLOv8 | 8.2.0 |
| Dataset | HuggingFace COCO stream | 2.20.0 |
| API | FastAPI + uvicorn | 0.111.0 |
| Cache | Redis | 7 |
| Monitoring | Prometheus + Grafana | latest |
| Experiment Tracking | MLflow | 2.14.1 |
| UI | Streamlit | 1.37.0 |
| Charts | Plotly | 5.22.0 |
| Testing | pytest + httpx | 8.3.2 |
| CI/CD | GitHub Actions | — |
| Containers | Docker Compose | — |
| Model Storage | HuggingFace Hub | — |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health + loaded models |
| `POST` | `/classify` | Image classification (ResNet50 or MobileNet) |
| `POST` | `/detect` | Object detection (YOLOv8n) |
| `GET` | `/metrics` | Prometheus metrics (text/plain) |
| `GET` | `/drift/status` | KS-test drift status per class |

```bash
# Example classify call
curl -X POST http://localhost:8000/classify \
  -F "file=@image.jpg" \
  -F "model_name=resnet50" \
  -F "top_k=5"

# Example detect call
curl -X POST http://localhost:8000/detect \
  -F "file=@image.jpg"
```

---

## Running Tests

```bash
pip install -r requirements.txt
pytest -v
# 57 tests: health(8), classify(14), detect(11), drift(14), redis(10)
```

---

## Dataset

- **Source:** COCO 2017 via `detection-datasets/coco` on HuggingFace (streaming — no full download)
- **Subset:** 2,500 images (100/class × 25 classes)
- **Splits:** 70% train / 15% val / 15% test
- **CNN input:** 224×224 cropped objects, ImageNet normalization
- **YOLO input:** 640×640 full scenes with bounding box annotations

---

## Reproducing Training

Training runs on Google Colab T4. Convert notebooks with jupytext, then open in Colab:

```bash
jupytext --to notebook notebooks/04_train_classifier.py
jupytext --to notebook notebooks/05_yolo_training.py
```

Set `FAST_MODE = False` at the top of each notebook for full training.
Model weights upload automatically to HuggingFace Hub after training.

---

## Deploying to HuggingFace Spaces

```bash
# 1. Create a new Space at huggingface.co (Streamlit SDK)
# 2. Add your Space as a remote
git remote add spaces https://huggingface.co/spaces/YOUR_USERNAME/smartvision-ai

# 3. Push
git push spaces master
```

The Space uses `spaces.yaml`, `packages.txt`, and `requirements-hf.txt` automatically.
Add your `HF_TOKEN` and `HF_REPO_ID` as Space secrets for model weight loading.
