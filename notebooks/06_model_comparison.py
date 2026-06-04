# notebooks/06_model_comparison.py
# Section 7: Model Comparison + MLflow + Drift Baseline
# Runs locally on CPU -- no GPU, no Docker, no Colab.
#
# Outputs:
#   artifacts/classification/vgg16/metrics.json        (reconstructed)
#   artifacts/comparison/model_metrics.json
#   artifacts/comparison/accuracy_comparison.png
#   artifacts/comparison/size_speed_scatter.png
#   artifacts/comparison/yolo_per_class.png
#   artifacts/drift/training_confidence_baseline.json
#   artifacts/drift/baseline_scores/scores_*.npy       (22 files)
#   artifacts/mlflow_exports/run_ids.json
#   mlruns/mlflow.db

# %% [0] ── FAST_MODE ─────────────────────────────────────────────────────────
# ================================================================
FAST_MODE = False   # LOCAL — flip True for dev/testing
# False → CPU benchmark n=100 runs, full drift baseline on val split
# True  → CPU benchmark n=10  runs, skip drift baseline generation
# ================================================================

# %% [1] ── Imports + PROJECT_ROOT ────────────────────────────────────────────
import re
import gc
import json
import inspect
import logging
import sqlite3
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import mlflow
import mlflow.tracking

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().parent
    if not (PROJECT_ROOT / "config.py").exists():
        PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CLASSES,
    CLASS_TO_IDX,
    RANDOM_STATE,
    NUM_CLASSES,
    IMAGE_SIZE,
    ARTIFACTS_DIR,
    DATA_PROCESSED_DIR,
    DRIFT_BASELINE_PATH,
    COMPARISON_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_EXPERIMENT_YOLO,
    KS_MIN_SAMPLES_FOR_TEST,
)
from src.models.model_factory   import get_model
from src.models.base_classifier import load_model, benchmark_inference
from src.data.dataset           import SmartVisionDataset, create_stratified_split
from src.data.augmentor         import get_eval_transforms

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

print(f"PROJECT_ROOT : {PROJECT_ROOT}")
print(f"FAST_MODE    : {FAST_MODE}")
print(f"NUM_CLASSES  : {NUM_CLASSES}")
print(f"IMAGE_SIZE   : {IMAGE_SIZE}")


# ── Shared helpers ────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """JSON serialiser that handles numpy / torch types. Order is MANDATORY."""
    def default(self, obj):
        if isinstance(obj, torch.Tensor):               # Tensor FIRST
            o = obj.detach().cpu().numpy()
            if o.ndim == 0 or (o.ndim == 1 and o.size == 1):
                return round(float(o.flat[0]), 6)
            return o.tolist()
        if isinstance(obj, np.bool_):                   # BEFORE np.integer (subclass)
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return round(float(obj), 6)                 # Rule 17: 6dp in artifacts
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def safe_filename(s: str) -> str:
    """Filesystem-safe name for .npy files (all COCO names are safe, but be robust)."""
    return re.sub(r'[^\w\-]', '_', s)


def safe_accuracy(model_name: str, metrics_dict: dict) -> float:
    """Return test_accuracy as float, 0.0 for None/missing/non-numeric."""
    val = metrics_dict[model_name].get("test_accuracy")
    if val is None:
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        logger.warning(f"{model_name}: test_accuracy={val!r} not numeric, treating as 0.0")
        return 0.0


def _dump(path: Path, data: object) -> None:
    """Write JSON with NumpyEncoder. mkdir -p on parent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


# ── Verify benchmark_inference parameter name ─────────────────────────────────
sig = inspect.signature(benchmark_inference)
assert "image_size" in sig.parameters, (
    f"benchmark_inference params are {list(sig.parameters)}. "
    "Update Cell 4 to match the actual parameter name."
)
print("benchmark_inference signature OK -- 'image_size' parameter confirmed")


# %% [2] ── Create VGG16 metrics.json (if missing or invalid) ─────────────────

VGG16_METRICS = {
    "model":             "vgg16",
    "dataset_round":     1,
    "dataset_note":      (
        "Round 1 (100 img/class, 69 train/class). "
        "Other models used Round 2 (200 img/class, 140 train/class)."
    ),
    "test_accuracy":     0.595000,
    "test_precision":    None,      # not measured -- not derivable from accuracy on 22-class problem
    "test_recall":       None,
    "test_f1":           None,
    "val_accuracy":      None,      # not separately measured -- null, not copied from test_accuracy
    "inference_ms":      None,      # weights not saved; CPU latency benchmarked in Cell 4
    "model_size_mb":     527.800000,
    "epochs_trained":    20,
    "fast_mode":         False,
    "weights_available": False,
    "training_note": (
        "VGG16 Round 1: 59.5% test accuracy, 4100 params/img overfitting. "
        "Weights not saved (Colab training). "
        "precision/recall/F1/val_accuracy not measured -- null, not fabricated."
    ),
}

vgg_path = ARTIFACTS_DIR / "classification" / "vgg16" / "metrics.json"
vgg_path.parent.mkdir(parents=True, exist_ok=True)

should_write = True
if vgg_path.exists():
    try:
        existing = json.loads(vgg_path.read_text())
        if (existing.get("model") == "vgg16"
                and existing.get("test_accuracy") == 0.595000
                and existing.get("val_accuracy") is None):
            should_write = False
            print("vgg16/metrics.json already valid -- skipping write")
    except (json.JSONDecodeError, KeyError):
        logger.warning("vgg16/metrics.json exists but is invalid -- overwriting")

if should_write:
    _dump(vgg_path, VGG16_METRICS)
    print(f"Written: {vgg_path.relative_to(PROJECT_ROOT)}")


# %% [3] ── Load all metrics JSONs + YOLO schema validation ───────────────────

CNN_MODELS = ["vgg16", "mobilenet", "efficientnet", "resnet50"]

cnn_metrics: dict = {}
for name in CNN_MODELS:
    path = ARTIFACTS_DIR / "classification" / name / "metrics.json"
    with open(path) as f:
        cnn_metrics[name] = json.load(f)
    print(f"  Loaded {name}: test_accuracy={cnn_metrics[name].get('test_accuracy')}")

with open(ARTIFACTS_DIR / "detection" / "yolo_metrics.json") as f:
    yolo_metrics = json.load(f)

# Schema validation -- catches KeyError in Cell 6 before it writes partial artifacts
REQUIRED_YOLO_KEYS = [
    "map50", "map50_95", "precision", "recall",
    "epochs_trained", "num_classes", "imgsz", "per_class",
]
missing_yolo = [k for k in REQUIRED_YOLO_KEYS if k not in yolo_metrics]
if missing_yolo:
    raise KeyError(
        f"yolo_metrics.json missing required keys: {missing_yolo}. "
        f"Available: {list(yolo_metrics.keys())}"
    )
print(f"YOLO metrics OK: {len(yolo_metrics['per_class'])} classes, mAP50={yolo_metrics['map50']:.4f}")


# %% [4] ── CPU inference benchmark (all models) ───────────────────────────────
# Uses get_model(pretrained=False): exact training topology (same Dropout + head),
# no weight download. Latency = f(architecture), not weight values.

BENCH_N      = 10 if FAST_MODE else 100
BENCH_WARMUP = 5
device       = torch.device("cpu")
cpu_bench: dict[str, Optional[float]] = {}

print(f"\nCPU benchmark: n={BENCH_N}, warmup={BENCH_WARMUP}")
for name in CNN_MODELS:
    try:
        model = get_model(name, num_classes=NUM_CLASSES, pretrained=False)
        model.eval()
        ms = benchmark_inference(
            model, device, image_size=IMAGE_SIZE, n=BENCH_N, warmup=BENCH_WARMUP
        )
        cpu_bench[name] = round(ms, 4)
        print(f"  [OK] {name}: {ms:.2f} ms CPU")
        del model
        gc.collect()    # explicit GC between large models; CPU-only, no GPU cache
    except Exception as e:
        cpu_bench[name] = None
        logger.warning(f"  [WARN] {name} benchmark failed: {e} -- null in comparison JSON")

# YOLO CPU benchmark (includes NMS post-processing -- documented in JSON)
try:
    from ultralytics import YOLO as UltralyticsYOLO
    yolo_pt_path = ARTIFACTS_DIR / "detection" / "yolov8_smartvision.pt"
    yolo_model = UltralyticsYOLO(str(yolo_pt_path))
    dummy_np = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        yolo_model.predict(dummy_np, verbose=False)   # warmup
    _yolo_times = []
    for _ in range(BENCH_N):
        t0 = time.perf_counter()
        yolo_model.predict(dummy_np, verbose=False)
        _yolo_times.append((time.perf_counter() - t0) * 1000)
    yolo_cpu_ms: Optional[float] = round(float(np.mean(_yolo_times)), 4)
    print(f"  [OK] yolov8n: {yolo_cpu_ms:.2f} ms CPU (includes NMS)")
    del yolo_model
    gc.collect()
except Exception as e:
    yolo_cpu_ms = None
    logger.warning(f"  [WARN] YOLO benchmark failed: {e} -- null in comparison JSON")


# %% [5] ── Build unified comparison JSON ─────────────────────────────────────

champion = max(CNN_MODELS, key=lambda m: safe_accuracy(m, cnn_metrics))

comparison = {
    "schema_version":         "1.0",
    "generated_by":           "notebooks/06_model_comparison.py",
    "generated_at":           datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "champion_classifier":    champion,
    "champion_test_accuracy": round(safe_accuracy(champion, cnn_metrics), 6),
    "models": {
        name: {
            "test_accuracy":     cnn_metrics[name].get("test_accuracy"),
            "test_precision":    cnn_metrics[name].get("test_precision"),   # None for VGG16
            "test_recall":       cnn_metrics[name].get("test_recall"),
            "test_f1":           cnn_metrics[name].get("test_f1"),
            "val_accuracy":      cnn_metrics[name].get("val_accuracy"),     # None for VGG16
            "model_size_mb":     cnn_metrics[name].get("model_size_mb"),
            "cpu_inference_ms":  cpu_bench[name],
            "cpu_inference_note": (
                "Architecture-only benchmark (pretrained=False). "
                "Correct topology (same Dropout + head as training). "
                "Latency = f(architecture), not weight values."
            ),
            "epochs_trained":    cnn_metrics[name].get("epochs_trained"),
            "dataset_round":     cnn_metrics[name].get("dataset_round", 2),
            "weights_available": cnn_metrics[name].get("weights_available", True),
        }
        for name in CNN_MODELS
    },
    "detection": {
        "yolov8n": {
            "map50":                      round(yolo_metrics["map50"],     6),
            "map50_95":                   round(yolo_metrics["map50_95"],  6),
            "precision":                  round(yolo_metrics["precision"], 6),
            "recall":                     round(yolo_metrics["recall"],    6),
            "epochs_trained":             yolo_metrics["epochs_trained"],
            "cpu_inference_ms":           yolo_cpu_ms,
            "cpu_inference_includes_nms": True,
            "cpu_inference_note":         "Includes NMS post-processing. CNN values are forward-pass only.",
        }
    },
}

_dump(COMPARISON_PATH, comparison)
print(f"Saved comparison JSON: {COMPARISON_PATH.relative_to(PROJECT_ROOT)}")
print(f"  champion={champion} ({safe_accuracy(champion, cnn_metrics):.1%})")


# %% [6] ── Comparison charts ─────────────────────────────────────────────────
# All matplotlib (Plotly is reserved for Streamlit pages).
# No Unicode in labels -- Windows cp1252 rule.
# Colormap range fixed at 0.50-1.0 -- consistent across re-runs.

COMPARISON_DIR = ARTIFACTS_DIR / "comparison"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

# ── Chart 1: accuracy_comparison.png ─────────────────────────────────────────
sorted_models = sorted(
    CNN_MODELS,
    key=lambda m: safe_accuracy(m, cnn_metrics),
    reverse=True,
)
accs   = [safe_accuracy(m, cnn_metrics) for m in sorted_models]
norm   = mcolors.Normalize(vmin=0.50, vmax=1.0)
cmap_g = cm.Greens
colors = [cmap_g(norm(a)) for a in accs]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(sorted_models, accs, color=colors, edgecolor="white", linewidth=0.5)
for bar, m_name, acc in zip(bars, sorted_models, accs):
    rnd    = cnn_metrics[m_name].get("dataset_round", 2)
    sz     = cnn_metrics[m_name].get("model_size_mb")
    sz_str = f"{sz:.0f}MB" if sz is not None else "?MB"
    label  = f"{acc:.1%}  {sz_str}  R{rnd}"
    ax.text(
        bar.get_width() + 0.005,
        bar.get_y() + bar.get_height() / 2,
        label, va="center", fontsize=9,
    )
ax.set_xlabel("Test Accuracy")
ax.set_title("CNN Test Accuracy -- 22-Class COCO Subset")
ax.set_xlim(0.0, 0.85)
ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
plt.tight_layout()
plt.savefig(COMPARISON_DIR / "accuracy_comparison.png", dpi=150)
plt.close()
print("Saved: accuracy_comparison.png")

# ── Chart 2: size_speed_scatter.png ──────────────────────────────────────────
plot_data = [
    (m, comparison["models"][m]["cpu_inference_ms"],
     safe_accuracy(m, cnn_metrics),
     comparison["models"][m].get("model_size_mb") or 1.0)
    for m in CNN_MODELS
    if comparison["models"][m]["cpu_inference_ms"] is not None
]

if plot_data:
    fig, ax = plt.subplots(figsize=(8, 6))
    _COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    for i, (m_name, ms, acc, size_mb) in enumerate(plot_data):
        ax.scatter(
            ms, acc,
            s=size_mb / 2,          # bubble size proportional to model_size_mb
            color=_COLORS[i % len(_COLORS)],
            alpha=0.75,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        ax.annotate(
            m_name,
            (ms, acc),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
        )
    ax.set_xlabel("CPU Inference (ms, architecture-only)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Model Size vs Speed vs Accuracy (CPU)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "size_speed_scatter.png", dpi=150)
    plt.close()
    print("Saved: size_speed_scatter.png")
else:
    logger.warning("No valid cpu_inference_ms values -- skipping scatter chart")

# ── Chart 3: yolo_per_class.png ──────────────────────────────────────────────
per_class_items = sorted(
    [(cls, v.get("ap50", 0.0)) for cls, v in yolo_metrics["per_class"].items()],
    key=lambda kv: kv[1],
    reverse=True,
)
cls_names  = [kv[0] for kv in per_class_items]
ap50_vals  = [kv[1] for kv in per_class_items]

fig, ax = plt.subplots(figsize=(10, 8))
y_pos = range(len(cls_names))
bar_colors = [cmap_g(norm(v)) for v in ap50_vals]
ax.barh(list(y_pos), ap50_vals, color=bar_colors, edgecolor="white", linewidth=0.4)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(cls_names, fontsize=8)
ax.set_xlabel("AP50")
ax.set_title("YOLOv8n Per-Class AP50 -- 50 Epochs (22 Classes)")
ax.axvline(x=yolo_metrics["map50"], color="red", linestyle="--",
           linewidth=1.0, alpha=0.7, label=f"mAP50={yolo_metrics['map50']:.3f}")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(COMPARISON_DIR / "yolo_per_class.png", dpi=150)
plt.close()
print("Saved: yolo_per_class.png")


# %% [7] ── MLflow logging ────────────────────────────────────────────────────

# Absolute path -- avoids CWD-relative resolution issues
db_path = (PROJECT_ROOT / "mlruns" / "mlflow.db").resolve()
db_path.parent.mkdir(parents=True, exist_ok=True)

# WAL mode for concurrent read/write (mlflow ui + notebook simultaneously)
# conn.commit() is mandatory -- PRAGMA is a write operation; must persist
conn = sqlite3.connect(str(db_path))
conn.execute("PRAGMA journal_mode=WAL;")
conn.commit()
conn.close()

mlflow.set_tracking_uri(f"sqlite:///{db_path}")

# Connectivity check before any runs
try:
    _client = mlflow.tracking.MlflowClient()
    _client.search_experiments()
    print(f"MLflow tracking confirmed: {db_path}")
except Exception as _e:
    raise RuntimeError(
        f"MLflow tracking store not accessible at {db_path}: {_e}. "
        "Check that mlruns/ directory exists and mlflow.db is not locked."
    ) from _e


def log_artifact_if_exists(path: Path, tag_key: str) -> None:
    """Log artifact only if the file exists -- missing PNG becomes a tag, not a crash."""
    if path.exists():
        mlflow.log_artifact(str(path))
    else:
        mlflow.set_tag(f"missing_{tag_key}", str(path))
        logger.warning(f"Artifact not found, skipping: {path}")


RUN_IDS_PATH = ARTIFACTS_DIR / "mlflow_exports" / "run_ids.json"
RUN_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
run_ids: dict = {}

# ── Experiment 1: CNN classification ─────────────────────────────────────────
clf_exp = mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
print(f"\nMLflow experiment: {MLFLOW_EXPERIMENT_NAME} (id={clf_exp.experiment_id})")

for name in CNN_MODELS:
    m = cnn_metrics[name]
    try:
        with mlflow.start_run(run_name=name, experiment_id=clf_exp.experiment_id) as run:
            mlflow.log_params({
                "model":          name,
                "num_classes":    int(NUM_CLASSES),
                "epochs_trained": int(m.get("epochs_trained") or 0),
                "dataset_round":  int(m.get("dataset_round", 2)),
                "fast_mode":      str(m.get("fast_mode", False)),
            })
            # Only log non-null metrics (VGG16 has null precision/recall/F1/val_accuracy)
            metrics_to_log = {
                k: round(float(v), 6)
                for k, v in {
                    "test_accuracy":    m.get("test_accuracy"),
                    "test_precision":   m.get("test_precision"),
                    "test_recall":      m.get("test_recall"),
                    "test_f1":          m.get("test_f1"),
                    "val_accuracy":     m.get("val_accuracy"),
                    "model_size_mb":    m.get("model_size_mb"),
                    "cpu_inference_ms": cpu_bench.get(name),
                }.items()
                if v is not None   # float("nan") also excluded by this filter
            }
            mlflow.log_metrics(metrics_to_log)
            mlflow.set_tag("weights_available", str(m.get("weights_available", True)))
            mlflow.set_tag("metrics_complete",  str(m.get("test_precision") is not None))

            clf_dir = ARTIFACTS_DIR / "classification" / name
            log_artifact_if_exists(clf_dir / "confusion_matrix.png",  "confusion_matrix")
            log_artifact_if_exists(clf_dir / "training_history.png",  "training_history")
            log_artifact_if_exists(clf_dir / "metrics.json",          "metrics_json")

            run_ids[name] = run.info.run_id
            print(f"  [OK] {name}: run_id={run.info.run_id[:8]}...")

    except Exception as e:
        logger.error(f"MLflow run FAILED for {name}: {e}")
        run_ids[f"{name}_ERROR"] = str(e)

    # Write incrementally after every CNN run -- safe if a subsequent run fails
    with open(RUN_IDS_PATH, "w") as f:
        json.dump(run_ids, f, indent=2)

# ── Experiment 2: YOLO detection ─────────────────────────────────────────────
det_exp = mlflow.set_experiment(MLFLOW_EXPERIMENT_YOLO)
print(f"\nMLflow experiment: {MLFLOW_EXPERIMENT_YOLO} (id={det_exp.experiment_id})")

try:
    with mlflow.start_run(run_name="yolov8n", experiment_id=det_exp.experiment_id) as run:
        mlflow.log_params({
            "model":       "yolov8n",
            "epochs":      int(yolo_metrics["epochs_trained"]),
            "num_classes": int(yolo_metrics["num_classes"]),
            "imgsz":       int(yolo_metrics["imgsz"]),
        })
        _yolo_mlflow_metrics = {
            "map50":     round(yolo_metrics["map50"],     6),
            "map50_95":  round(yolo_metrics["map50_95"],  6),
            "precision": round(yolo_metrics["precision"], 6),
            "recall":    round(yolo_metrics["recall"],    6),
        }
        if yolo_cpu_ms is not None:
            _yolo_mlflow_metrics["cpu_inference_ms"] = float(yolo_cpu_ms)
        mlflow.log_metrics(_yolo_mlflow_metrics)

        det_dir = ARTIFACTS_DIR / "detection"
        log_artifact_if_exists(det_dir / "confusion_matrix.png", "confusion_matrix")
        log_artifact_if_exists(det_dir / "training_curves.png",  "training_curves")
        log_artifact_if_exists(det_dir / "yolo_metrics.json",    "metrics_json")

        run_ids["yolov8n"] = run.info.run_id
        print(f"  [OK] yolov8n: run_id={run.info.run_id[:8]}...")

except Exception as e:
    logger.error(f"MLflow YOLO run FAILED: {e}")
    run_ids["yolov8n_ERROR"] = str(e)

# Write after YOLO block (consistent with per-CNN write pattern above)
with open(RUN_IDS_PATH, "w") as f:
    json.dump(run_ids, f, indent=2)
print(f"MLflow logging complete. Run IDs: {RUN_IDS_PATH.relative_to(PROJECT_ROOT)}")


# %% [8] ── Drift baseline ────────────────────────────────────────────────────
# Split   : val (held-out, not seen during training -- representative of deployment)
# Model   : MobileNet (only model with local .pt weights)
# Transform: get_eval_transforms() from src/data/augmentor.py -- same as training
# Scores  : max-softmax per sample, grouped by TRUE class
# Storage : summary stats in JSON, raw scores in .npy files (not in JSON)
# Accuracy: derived from cnn_metrics["mobilenet"] -- not hardcoded

if FAST_MODE:
    print("\nFAST_MODE: skipping drift baseline. Run with FAST_MODE=False to generate.")
else:
    print("\nGenerating drift baseline (val split, MobileNet)...")

    model = get_model("mobilenet", num_classes=NUM_CLASSES)   # pretrained=True (loads weights)
    model = load_model(
        model,
        PROJECT_ROOT / "models" / "mobilenet_best.pt",
        device="cpu",
    )   # weights_only=True inside load_model -- Rule 36
    model.eval()

    # Import transform from augmentor -- mandatory (NOT hardcoded here)
    # get_eval_transforms: ToImage + ToDtype + Resize(224,224) -- no CenterCrop
    eval_transform = get_eval_transforms(image_size=IMAGE_SIZE)

    data_root = DATA_PROCESSED_DIR / "classification"
    splits    = create_stratified_split(data_root, random_state=RANDOM_STATE)
    val_ds    = SmartVisionDataset("val", transform=eval_transform, samples=splits["val"])
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=32, shuffle=False, num_workers=0
    )
    n_per_class = len(val_ds) // NUM_CLASSES
    print(f"  Val split: {len(val_ds)} images (~{n_per_class} per class)")

    # Collect confidence scores: max-softmax + accuracy-conditional separation
    conf_all:       dict[str, list[float]] = {cls: [] for cls in CLASSES}
    conf_correct:   dict[str, list[float]] = {cls: [] for cls in CLASSES}
    conf_incorrect: dict[str, list[float]] = {cls: [] for cls in CLASSES}

    with torch.no_grad():
        for images, labels in val_loader:
            probs       = torch.softmax(model(images), dim=1)
            confidences = probs.max(dim=1).values
            preds       = probs.argmax(dim=1)
            for conf, pred, label in zip(confidences, preds, labels):
                cls_name = CLASSES[label.item()]
                c        = conf.item()
                conf_all[cls_name].append(c)
                if pred.item() == label.item():
                    conf_correct[cls_name].append(c)
                else:
                    conf_incorrect[cls_name].append(c)

    # Save raw scores as .npy (not embedded in JSON)
    scores_dir = DRIFT_BASELINE_PATH.parent / "baseline_scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    for cls_name, scores in conf_all.items():
        np.save(
            scores_dir / f"scores_{safe_filename(cls_name)}.npy",
            np.array(scores, dtype=np.float32),
        )

    # Derive model_accuracy from loaded JSON -- never hardcode
    mobilenet_accuracy = cnn_metrics["mobilenet"].get("test_accuracy")

    baseline_summary = {
        "schema_version":      "1.0",
        "model_used":          "mobilenet",
        "model_accuracy":      round(float(mobilenet_accuracy), 6) if mobilenet_accuracy else None,
        "split_used":          "val",
        "random_state":        int(RANDOM_STATE),
        "eval_transform":      "get_eval_transforms from src/data/augmentor.py (ToImage+ToDtype+Resize224)",
        "note": (
            "MobileNet used (only model with local weights). "
            "Val split: held-out, not seen during training, representative of deployment. "
            "Update with ResNet50 (champion, 65.5%) after HF model upload in Section 12. "
            "Known limitation: max-softmax from a 56.7% model is overconfident on errors. "
            "Accuracy-conditional distributions (mean_correct, mean_incorrect) provided "
            "to mitigate this for Section 9 KS test."
        ),
        "scores_base_dir":     "artifacts/drift/baseline_scores",
        "scores_file_note": (
            "scores_file is relative to this JSON file's directory (artifacts/drift/). "
            "scores_base_dir is relative to PROJECT_ROOT for programmatic access."
        ),
        "ks_min_samples":      int(KS_MIN_SAMPLES_FOR_TEST),
        "n_samples_per_class": int(n_per_class),
        "classes": {
            cls: {
                "mean":           round(float(np.mean(scores)), 6) if scores else None,
                "std":            round(float(np.std(scores)),  6) if scores else None,
                "n_samples":      int(len(scores)),
                "mean_correct":   round(float(np.mean(conf_correct[cls])), 6) if conf_correct[cls] else None,
                "n_correct":      int(len(conf_correct[cls])),
                "mean_incorrect": round(float(np.mean(conf_incorrect[cls])), 6) if conf_incorrect[cls] else None,
                "n_incorrect":    int(len(conf_incorrect[cls])),
                "scores_file":    f"baseline_scores/scores_{safe_filename(cls)}.npy",
            }
            for cls, scores in conf_all.items()
        },
    }

    _dump(DRIFT_BASELINE_PATH, baseline_summary)
    n_npys = len(list(scores_dir.glob("*.npy")))
    print(f"  Drift baseline saved: {DRIFT_BASELINE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  {n_npys} .npy score files in {scores_dir.relative_to(PROJECT_ROOT)}")


# %% [9] ── Validation ────────────────────────────────────────────────────────

def validate_artifact(path: Path, expected_keys: Optional[list] = None) -> bool:
    """Validate file exists, is non-empty, and (if JSON) has required keys."""
    if not path.exists():
        print(f"  [FAIL] Missing: {path.relative_to(PROJECT_ROOT)}")
        return False
    if path.stat().st_size == 0:
        print(f"  [FAIL] Empty file: {path.relative_to(PROJECT_ROOT)}")
        return False
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text())
            if expected_keys:
                missing = [k for k in expected_keys if k not in data]
                if missing:
                    print(f"  [FAIL] {path.name}: missing keys {missing}")
                    return False
        except json.JSONDecodeError as e:
            print(f"  [FAIL] Invalid JSON at {path.name}: {e}")
            return False
    print(f"  [OK]   {path.relative_to(PROJECT_ROOT)}")
    return True


print("\nValidating artifacts...")

# Always validate drift baseline if it exists -- regardless of FAST_MODE
drift_check = True
if DRIFT_BASELINE_PATH.exists():
    drift_check = validate_artifact(
        DRIFT_BASELINE_PATH,
        ["model_used", "split_used", "classes", "random_state", "eval_transform"],
    )
    if drift_check:
        _d = json.loads(DRIFT_BASELINE_PATH.read_text())
        mob_acc = _d.get("model_accuracy")
        assert mob_acc is not None, "model_accuracy must not be None"
        assert isinstance(mob_acc, (int, float)), f"model_accuracy must be numeric, got {type(mob_acc)}"
        assert 0.0 < mob_acc < 1.0, f"model_accuracy={mob_acc} outside (0, 1)"
        assert _d.get("split_used") == "val", f"split_used must be 'val', got {_d.get('split_used')}"
        print(f"         model_accuracy={mob_acc:.4f} (from metrics.json, not hardcoded)")
        print(f"         split_used={_d['split_used']} -- confirmed val, not train")
        print(f"         n_classes={len(_d['classes'])}")

checks = [
    validate_artifact(
        ARTIFACTS_DIR / "classification" / "vgg16" / "metrics.json",
        ["model", "test_accuracy", "model_size_mb"],
    ),
    validate_artifact(
        COMPARISON_PATH,
        ["schema_version", "champion_classifier", "models", "detection"],
    ),
    validate_artifact(ARTIFACTS_DIR / "mlflow_exports" / "run_ids.json"),
    validate_artifact(COMPARISON_DIR / "accuracy_comparison.png"),
    validate_artifact(COMPARISON_DIR / "size_speed_scatter.png"),
    validate_artifact(COMPARISON_DIR / "yolo_per_class.png"),
    drift_check,
]

assert all(checks), "Validation FAILED -- see [FAIL] lines above"
print("\nAll artifacts validated. Section 7 complete.")

# Summary table
print(f"\n{'Model':<14} {'TestAcc':>8} {'SizeMB':>8} {'CPU_ms':>8} {'Round':>6}")
print("-" * 50)
for name, d in comparison["models"].items():
    acc = f"{safe_accuracy(name, cnn_metrics):.1%}"
    sz  = f"{d['model_size_mb']:.1f}"    if d["model_size_mb"]    is not None else "N/A"
    ms  = f"{d['cpu_inference_ms']:.1f}" if d["cpu_inference_ms"] is not None else "N/A"
    rnd = str(d.get("dataset_round", "?"))
    print(f"{name:<14} {acc:>8} {sz:>8} {ms:>8} {rnd:>6}")

print(f"\nchampion : {comparison['champion_classifier']} "
      f"({comparison['champion_test_accuracy']:.1%})")
print(f"MLflow DB: {db_path}")
print(f"Run:  mlflow ui --backend-store-uri sqlite:///{db_path}")
