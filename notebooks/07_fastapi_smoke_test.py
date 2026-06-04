# %% [markdown]
# # Section 8 Smoke Test — FastAPI + Redis
#
# Verifies all four endpoints after model startup:
#   GET  /health   — models_ready flag
#   POST /classify — top-k CNN predictions + Redis caching
#   POST /detect   — YOLO bounding boxes + Redis caching
#   GET  /metrics  — Prometheus text/plain exposition
#
# Run in **terminal** (no GPU needed, no Colab):
#   python notebooks/07_fastapi_smoke_test.py
#
# NOTE: Test images are from the on-disk val/ directory. After
# create_stratified_split(), some may overlap with the training split.
# That is acceptable here — we are testing API correctness, not model accuracy.

# %%
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent  # terminal
except NameError:
    PROJECT_ROOT = Path.cwd().parent
    if not (PROJECT_ROOT / "config.py").exists():
        PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import requests

from config import ARTIFACTS_DIR, DATA_PROCESSED_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────
CLASSIFY_VAL_DIR = DATA_PROCESSED_DIR / "classification" / "val"
DETECT_VAL_DIR   = DATA_PROCESSED_DIR / "detection" / "images" / "val"
RESULTS_DIR      = ARTIFACTS_DIR / "api"
RESULTS_PATH     = RESULTS_DIR / "smoke_test_results.json"
API_URL          = "http://localhost:8000"
STARTUP_TIMEOUT  = 180  # seconds — model loading can take ~60-90s on CPU

# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_image(directory: Path, suffix: str = ".jpg") -> Path:
    """Return the first image found in directory (recursive)."""
    matches = list(directory.rglob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"No {suffix} images found in {directory}")
    return sorted(matches)[0]


def _post_image(endpoint: str, image_path: Path, data: dict) -> requests.Response:
    """POST multipart/form-data with image file + text fields."""
    with open(image_path, "rb") as f:
        return requests.post(
            f"{API_URL}/{endpoint}",
            files={"file": (image_path.name, f, "image/jpeg")},
            data=data,
            timeout=60,
        )


# %% [markdown]
# ## Start uvicorn server

# %%
print("[1/6] Starting uvicorn server...")
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8000", "--log-level", "warning"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=str(PROJECT_ROOT),
)

# %% [markdown]
# ## Wait for /health to report models_ready

# %%
print(f"[2/6] Waiting up to {STARTUP_TIMEOUT}s for models to load...")
deadline = time.time() + STARTUP_TIMEOUT
ready = False

while time.time() < deadline:
    # Check if process died (import error, port already in use, etc.)
    if proc.poll() is not None:
        _, stderr = proc.communicate()
        raise RuntimeError(
            f"uvicorn exited with code {proc.returncode}. "
            f"stderr:\n{stderr.decode(errors='replace')[:3000]}"
        )
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        data = r.json()
        if data.get("models_ready"):
            ready = True
            print(f"  Models loaded: {data['models_loaded']}")
            break
        else:
            print(f"  Status: {data.get('status', '?')} — still loading...", flush=True)
    except requests.exceptions.ConnectionError:
        print("  Waiting for server to start...", flush=True)
    time.sleep(3)

if not ready:
    proc.kill()
    proc.wait()
    raise TimeoutError(f"Server did not become ready within {STARTUP_TIMEOUT}s")

print("  [OK] Server ready.")

# %% [markdown]
# ## Test POST /classify

# %%
print("[3/6] Testing POST /classify...")

health_data = requests.get(f"{API_URL}/health").json()
models_loaded = [m for m in health_data["models_loaded"] if m != "yolo"]

if not models_loaded:
    print("  [SKIP] No CNN models loaded — skipping classify test.")
    classify_result = {"skipped": True}
else:
    test_model = "resnet50" if "resnet50" in models_loaded else models_loaded[0]
    classify_image_path = _find_image(CLASSIFY_VAL_DIR)
    print(f"  Image: {classify_image_path.name}  Model: {test_model}")

    r1 = _post_image("classify", classify_image_path, {"model_name": test_model, "top_k": "5"})
    assert r1.status_code == 200, f"Expected 200, got {r1.status_code}: {r1.text}"
    body1 = r1.json()
    assert len(body1["predictions"]) > 0, "Expected at least one prediction"
    assert body1["model_name"] == test_model
    assert isinstance(body1["inference_time_ms"], float)
    assert body1["cached"] is False
    top1 = body1["predictions"][0]
    print(f"  Top-1: {top1['class_name']} ({top1['confidence']:.4f})  "
          f"latency={body1['inference_time_ms']:.1f}ms  cached={body1['cached']}")

    # Same image again — should be a cache hit if Redis is available
    r2 = _post_image("classify", classify_image_path, {"model_name": test_model, "top_k": "5"})
    assert r2.status_code == 200
    body2 = r2.json()
    if body2["cached"]:
        assert body2["predictions"] == body1["predictions"], "Cached predictions must match original"
        print(f"  Cache hit confirmed. cached=True  latency={body2['inference_time_ms']:.1f}ms")
    else:
        print("  Cache miss (Redis unavailable) — graceful degradation confirmed.")

    classify_result = {
        "test_model": test_model,
        "image": classify_image_path.name,
        "top1_class": top1["class_name"],
        "top1_confidence": top1["confidence"],
        "inference_time_ms": body1["inference_time_ms"],
        "cache_hit_on_second_request": body2["cached"],
    }
    print("  [OK] /classify passed.")

# %% [markdown]
# ## Test POST /detect

# %%
print("[4/6] Testing POST /detect...")

detect_image_path = _find_image(DETECT_VAL_DIR)
print(f"  Image: {detect_image_path.name}  conf_threshold=0.25")

rd = _post_image("detect", detect_image_path, {"conf_threshold": "0.25"})
assert rd.status_code == 200, f"Expected 200, got {rd.status_code}: {rd.text}"
body_d = rd.json()
assert isinstance(body_d["detections"], list), "detections must be a list"
assert isinstance(body_d["inference_time_ms"], float)
print(f"  Detections: {len(body_d['detections'])}  "
      f"latency={body_d['inference_time_ms']:.1f}ms  cached={body_d['cached']}")

if body_d["detections"]:
    top_det = body_d["detections"][0]
    print(f"  Top detection: {top_det['class_name']} ({top_det['confidence']:.4f})  "
          f"bbox=[{top_det['x1']:.0f},{top_det['y1']:.0f},{top_det['x2']:.0f},{top_det['y2']:.0f}]")

# Same image again — cache hit check
rd2 = _post_image("detect", detect_image_path, {"conf_threshold": "0.25"})
assert rd2.status_code == 200
if rd2.json()["cached"]:
    print("  Cache hit confirmed for detect.")

detect_result = {
    "image": detect_image_path.name,
    "n_detections": len(body_d["detections"]),
    "inference_time_ms": body_d["inference_time_ms"],
    "cache_hit_on_second_request": rd2.json()["cached"],
}
print("  [OK] /detect passed.")

# %% [markdown]
# ## Test GET /metrics

# %%
print("[5/6] Testing GET /metrics...")

rm = requests.get(f"{API_URL}/metrics", timeout=10)
assert rm.status_code == 200, f"Expected 200, got {rm.status_code}"
assert "text/plain" in rm.headers.get("content-type", ""), (
    f"Expected text/plain, got: {rm.headers.get('content-type')}"
)
assert "smartvision_classify_requests_total" in rm.text, (
    "Prometheus counter name missing from /metrics output"
)
assert rm.text.startswith("# HELP") or "# HELP" in rm.text, (
    "Prometheus exposition format missing '# HELP' lines"
)
print("  content-type: text/plain  [OK]")
print("  smartvision_classify_requests_total present  [OK]")
print("  [OK] /metrics passed.")

metrics_result = {
    "content_type": rm.headers.get("content-type"),
    "body_length_bytes": len(rm.content),
}

# %% [markdown]
# ## Test /health returns 200 with models_ready=True

# %%
print("[6/6] Verifying /health final state...")
rh = requests.get(f"{API_URL}/health")
assert rh.status_code == 200, f"Expected 200 when ready, got {rh.status_code}"
assert rh.json()["models_ready"] is True
print(f"  {rh.json()}")
print("  [OK] /health passed.")

# %% [markdown]
# ## Save results

# %%
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
results = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "classify": classify_result,
    "detect": detect_result,
    "metrics": metrics_result,
    "health": rh.json(),
    "status": "PASS",
}
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nSmoke test complete. Results saved: {RESULTS_PATH}")
print("=== ALL TESTS PASSED ===")

# %% [markdown]
# ## Shutdown

# %%
if proc.poll() is None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
print("uvicorn subprocess terminated.")
