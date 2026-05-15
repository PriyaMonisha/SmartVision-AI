# Rule: No Model Calls in Pages

Pages are display-only. They call FastAPI endpoints — they do NOT load or run models.

**Forbidden in any `pages/` file:**
- `model(x)` or `model.forward(x)`
- `torch.load(...)` for model weights
- `YOLO(...)` instantiation
- Any import from `src/models/`

**Allowed:**
- `requests.post(f"{FASTAPI_URL}/classify", ...)`
- `requests.post(f"{FASTAPI_URL}/detect", ...)`
- `load_json(ARTIFACTS_DIR / "comparison" / "model_metrics.json")`

Also: `src/` modules must NEVER import from `api/`. Use dependency injection.
