# filename: config.py
# purpose:  Central configuration — all constants, paths, hyperparameters for SmartVision AI
# version:  2.0

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

FAST_MODE   = settings.fast_mode
FASTAPI_URL = settings.fastapi_url
HF_TOKEN    = settings.hf_token
HF_REPO_ID  = settings.hf_repo_id
REDIS_HOST  = settings.redis_host
REDIS_PORT  = settings.redis_port

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Model dimensions ─────────────────────────────────────────────────────────
IMAGE_SIZE      = 224
YOLO_IMAGE_SIZE = 640
NUM_CLASSES     = 25

# ── Class definitions ─────────────────────────────────────────────────────────
# Source of truth for class ordering (0-24).
# ORDER matches mentor's notebook (by ascending HF category ID).
# Mentor notebook included 'train' (vehicle) making it 26 — we exclude it (project PDF = 25).
CLASSES = [
    "person",        # 0
    "bicycle",       # 1
    "car",           # 2
    "motorcycle",    # 3
    "airplane",      # 4
    "bus",           # 5
    "truck",         # 6  (HF cat 7 — skipping 'train' at HF cat 6)
    "traffic light", # 7
    "stop sign",     # 8
    "bench",         # 9
    "bird",          # 10
    "cat",           # 11
    "dog",           # 12
    "horse",         # 13
    "cow",           # 14
    "elephant",      # 15
    "bottle",        # 16
    "cup",           # 17
    "bowl",          # 18
    "pizza",         # 19
    "cake",          # 20
    "chair",         # 21
    "couch",         # 22
    "potted plant",  # 23
    "bed",           # 24
]

# Maps class name → HuggingFace detection-datasets/coco 0-indexed category ID.
# The HF dataset uses 0-indexed IDs (0-79), NOT original COCO annotation IDs (1-90 with gaps).
# This is the primary lookup used during streaming in 01_data_acquisition.py.
SELECTED_CLASSES: dict[str, int] = {
    "person":        0,
    "bicycle":       1,
    "car":           2,
    "motorcycle":    3,
    "airplane":      4,
    "bus":           5,
    "truck":         7,   # HF ID 7 — HF ID 6 = 'train' (vehicle), excluded
    "traffic light": 9,
    "stop sign":     11,
    "bench":         13,
    "bird":          14,
    "cat":           15,
    "dog":           16,
    "horse":         17,
    "cow":           19,
    "elephant":      20,
    "bottle":        39,
    "cup":           41,
    "bowl":          45,
    "pizza":         53,
    "cake":          55,
    "chair":         56,
    "couch":         57,
    "potted plant":  58,
    "bed":           59,
}

# Reverse mapping: HF category ID → our sequential class index (0-24).
# Used when iterating annotations to assign YOLO class IDs.
HF_CATEGORY_TO_CLASS_IDX: dict[int, int] = {
    hf_id: CLASSES.index(cls_name)
    for cls_name, hf_id in SELECTED_CLASSES.items()
}

CLASS_TO_IDX: dict[str, int] = {cls: idx for idx, cls in enumerate(CLASSES)}

# ── Dataset splits ────────────────────────────────────────────────────────────
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.70, 0.15, 0.15
IMAGES_PER_CLASS      = 100
FAST_IMAGES_PER_CLASS = 10

# ── Per-model training config ─────────────────────────────────────────────────
# VGG16 batch=16: model ~550MB, batch=32 is borderline on T4 (memory budget)
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
FASTAPI_HOST              = "0.0.0.0"
FASTAPI_PORT              = 8000
INFERENCE_TIMEOUT_SECONDS = 30

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI    = "sqlite:///mlruns/mlflow.db"
MLFLOW_EXPERIMENT_NAME = "smartvision_classification"
MLFLOW_EXPERIMENT_YOLO = "smartvision_detection"

# ── Drift detection ───────────────────────────────────────────────────────────
KS_DRIFT_ALERT_THRESHOLD = 0.10
KS_MIN_SAMPLES_FOR_TEST  = 100

# ── Streamlit input validation ────────────────────────────────────────────────
MAX_IMAGE_SIZE_MB    = 10
MAX_IMAGE_RESOLUTION = 1920

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_TTL_CLASSIFY = 86400   # 24h
REDIS_TTL_DETECT   = 3600    # 1h

# ── Memory budget (documented for Docker mem_limit) ───────────────────────────
# VGG16:          ~550MB | ResNet50: ~100MB | MobileNetV2: ~14MB
# EfficientNetB0: ~20MB  | YOLOv8n:  ~6MB
# Total FastAPI:  ~690MB → Docker mem_limit: 1.5g

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
