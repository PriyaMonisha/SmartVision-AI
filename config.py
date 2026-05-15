# filename: config.py
# purpose:  Central configuration — all constants, paths, hyperparameters for SmartVision AI
# version:  1.0

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    hf_token: str = ""
    hf_repo_id: str = "your-username/smartvision-models"
    fastapi_url: str = "http://localhost:8000"
    redis_host: str = "redis"
    redis_port: int = 6379
    # FAST_MODE reads from env var for Docker/production.
    # In notebooks: override as LOCAL variable, pass as function param (never mutate this).
    fast_mode: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Re-export as module-level names for backward compatibility
FAST_MODE   = settings.fast_mode
FASTAPI_URL = settings.fastapi_url
HF_TOKEN    = settings.hf_token
HF_REPO_ID  = settings.hf_repo_id
REDIS_HOST  = settings.redis_host
REDIS_PORT  = settings.redis_port

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Model dimensions ─────────────────────────────────────────────────────────
IMAGE_SIZE      = 224   # CNN input size (px)
YOLO_IMAGE_SIZE = 640   # YOLO input size (px)
NUM_CLASSES     = 25

# ── 25 Classes — GROUND TRUTH ordering (verified against COCO 2017) ──────────
# Index = class_id used by both CNN classifiers and YOLO labels
# Comments show COCO category_id for cross-reference and auditing
CLASSES = [
    "person",        # 0  — COCO ID: 1
    "bicycle",       # 1  — COCO ID: 2
    "car",           # 2  — COCO ID: 3
    "motorcycle",    # 3  — COCO ID: 4
    "airplane",      # 4  — COCO ID: 5
    "bus",           # 5  — COCO ID: 6
    "truck",         # 6  — COCO ID: 8
    "bird",          # 7  — COCO ID: 16
    "cat",           # 8  — COCO ID: 17
    "dog",           # 9  — COCO ID: 18
    "horse",         # 10 — COCO ID: 19
    "cow",           # 11 — COCO ID: 21
    "elephant",      # 12 — COCO ID: 22
    "bench",         # 13 — COCO ID: 15
    "traffic light", # 14 — COCO ID: 10
    "stop sign",     # 15 — COCO ID: 13
    "bottle",        # 16 — COCO ID: 44
    "cup",           # 17 — COCO ID: 47
    "bowl",          # 18 — COCO ID: 51
    "chair",         # 19 — COCO ID: 62
    "couch",         # 20 — COCO ID: 63
    "potted plant",  # 21 — COCO ID: 64
    "bed",           # 22 — COCO ID: 65
    "pizza",         # 23 — COCO ID: 59
    "cake",          # 24 — COCO ID: 61
]

# Maps COCO annotation category_id → our 0-24 class index
# Used by loader.py for both classification crops and YOLO label generation
EXPECTED_COCO_CATEGORIES = {
    1: "person",        2: "bicycle",      3: "car",         4: "motorcycle",
    5: "airplane",      6: "bus",          8: "truck",
    10: "traffic light", 13: "stop sign",  15: "bench",
    16: "bird",         17: "cat",         18: "dog",        19: "horse",
    21: "cow",          22: "elephant",
    44: "bottle",       47: "cup",         51: "bowl",
    59: "pizza",        61: "cake",
    62: "chair",        63: "couch",       64: "potted plant", 65: "bed",
}

COCO_ID_TO_CLASS_IDX: dict[int, int] = {
    1: 0,   2: 1,   3: 2,   4: 3,   5: 4,   6: 5,   8: 6,
    16: 7,  17: 8,  18: 9,  19: 10, 21: 11, 22: 12,
    15: 13, 10: 14, 13: 15,
    44: 16, 47: 17, 51: 18,
    62: 19, 63: 20, 64: 21, 65: 22,
    59: 23, 61: 24,
}

CLASS_TO_IDX: dict[str, int] = {cls: idx for idx, cls in enumerate(CLASSES)}

# ── Dataset splits ────────────────────────────────────────────────────────────
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.70, 0.15, 0.15
IMAGES_PER_CLASS      = 100
FAST_IMAGES_PER_CLASS = 10   # dev runs

# ── Per-model training config ─────────────────────────────────────────────────
# VGG16 batch=16: model ~550MB, batch=32 is borderline on T4 (Rule 25)
MODEL_CONFIGS: dict[str, dict] = {
    "vgg16":        {"lr": 0.001,  "epochs": 20, "batch": 16, "unfreeze": "none"},
    "resnet50":     {"lr": 0.0001, "epochs": 25, "batch": 32, "unfreeze": "layer3+"},
    "mobilenet":    {"lr": 0.001,  "epochs": 20, "batch": 64, "unfreeze": "none"},
    "efficientnet": {"lr": 0.0001, "epochs": 25, "batch": 32, "unfreeze": "none"},
}

# ── YOLO ──────────────────────────────────────────────────────────────────────
YOLO_EPOCHS         = 50
YOLO_BATCH          = 16
YOLO_CONF_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD  = 0.45

# ── FastAPI / serving ─────────────────────────────────────────────────────────
FASTAPI_HOST               = "0.0.0.0"
FASTAPI_PORT               = 8000
INFERENCE_TIMEOUT_SECONDS  = 30

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI        = "sqlite:///mlruns/mlflow.db"
MLFLOW_EXPERIMENT_NAME     = "smartvision_classification"
MLFLOW_EXPERIMENT_YOLO     = "smartvision_detection"

# ── Drift detection ───────────────────────────────────────────────────────────
KS_DRIFT_ALERT_THRESHOLD = 0.10
KS_MIN_SAMPLES_FOR_TEST  = 100   # minimum live samples before running KS

# ── Streamlit input validation ────────────────────────────────────────────────
MAX_IMAGE_SIZE_MB    = 10
MAX_IMAGE_RESOLUTION = 1920

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_TTL_CLASSIFY  = 86400   # 24h for classification results
REDIS_TTL_DETECT    = 3600    # 1h for detection results

# ── Memory budget (documented for Docker mem_limit) ───────────────────────────
# VGG16 classifier:          ~550MB
# ResNet50 classifier:       ~100MB
# MobileNetV2 classifier:    ~14MB
# EfficientNetB0 classifier: ~20MB
# YOLOv8n detector:          ~6MB
# Total FastAPI service:     ~690MB → Docker mem_limit: 1.5g

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR            = Path(__file__).parent
DATA_RAW_DIR        = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR  = BASE_DIR / "data" / "processed" / "smartvision_dataset"
ARTIFACTS_DIR       = BASE_DIR / "artifacts"
MODELS_DIR          = BASE_DIR / "models"
DOCS_FIGURES_DIR    = BASE_DIR / "docs" / "figures"

DRIFT_BASELINE_PATH = ARTIFACTS_DIR / "drift" / "training_confidence_baseline.json"
COMPARISON_PATH     = ARTIFACTS_DIR / "comparison" / "model_metrics.json"
CHECKPOINT_FILE     = DATA_PROCESSED_DIR / "download_progress.json"
