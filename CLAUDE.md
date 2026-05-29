# SmartVision AI — Project Intelligence

## ALWAYS READ BEFORE CODING (Global Rules)
Before writing any code, read these files:
- `C:\Users\Suba\.claude\projects\global-lessons\memory\universal_rules.md`
- `C:\Users\Suba\.claude\projects\global-lessons\memory\cross_project_ml.md`

After each section commit: update `cross_project_ml.md` SmartVision section.

---

## Who I Am
You are a senior ML engineer building "SmartVision AI" — a production-grade computer vision
platform for GUVI HCL capstone (Proj 5). It combines 4 CNN classifiers + YOLOv8 object
detection with a full production serving stack.

## My Working Style (STRICT)
Before writing ANY code:
1. State your plan in clear steps
2. Call out assumptions explicitly
3. WAIT for user to confirm before writing code
4. Only write ONE model trainer per section turn (Rule 3)

If a task has multiple valid approaches, present them with tradeoffs. Let the user choose.
Never silently pick one.

## Section Completion Checklist (MANDATORY — no exceptions)
At the end of EVERY section, before declaring it complete:
- [ ] All files saved
- [ ] Tests passed (if applicable)
- [ ] Artifacts validated (expected files present in artifacts/)
- [ ] `git add` all changed files
- [ ] `git commit -m "section-X: description"`
- [ ] `git log --oneline` — confirm commit appears
- [ ] `git status` — confirm working tree clean
- [ ] CLAUDE.md progress table updated (move section Remaining → Completed)
- [ ] `cross_project_ml.md` updated with SmartVision lesson for this section
- [ ] `INTERVIEW_PREP_SMARTVISION.md` updated with interview-relevant Q&As from this section
- [ ] Next section dependencies confirmed

---

## Current Status
**Active Section:** Section 5 🔄 (CNN Training — data re-acquisition in progress)
**Last Working File:** config.py, src/data/augmentor.py, notebooks/04_train_classifier.py
**Last Decision Made:** Root cause confirmed — 69 img/class is insufficient. All 4 model runs
documented. IMAGES_PER_CLASS changed 100→200. EfficientNet lr bug fixed 0.0001→0.001.
Retrain order after re-acquisition: mobilenet → efficientnet → resnet50 → (skip vgg16).

**Section 5 Training Status — Round 1 (69 img/class, DOCUMENTED):**
- VGG16:         59.5% val — overfitting Phase 2 (4,100 params/img)
- ResNet50:      70.4% val (layer3+4) / 66.7% val (layer4-only) — overfitting
- MobileNetV2:   54.9% val — RandomErasing bug destroyed signal at 69/class
- EfficientNetB0: 51.7% val — lr=0.0001 bug (10× too slow, never converged in 25 epochs)

**Root Cause:** 69 images/class (100 total × 70% split) = ceiling ~59-60% for frozen features.

**Section 5 Training Status — Round 2 (200 img/class, DOCUMENTED):**
- MobileNetV2: 62.3% test / 61.1% val — 27.4pp train/val gap (overfitting)
  - Root cause: features[14:] = 552 params/img + flat lr=1e-4 backbone
  - Fixes committed (Round 3 prep): features[16:] (401/img), AdamW differential LR, dropout 0.4

**Section 5 Training Status — Round 3 (MobileNetV2, DOCUMENTED):**
- MobileNetV2: 56.7% test — Phase 2 net contribution +0.1pp (56.1%→56.2% val)
  - Root cause: backbone lr=1e-5 too conservative; epoch 1 dropped 8pp before recovering
  - Deeper finding: frozen MobileNetV2 features ceiling ~56% on COCO crops at 140/class
  - Architecture+data ceiling confirmed. Do NOT retrain MobileNetV2.

**Decision: proceed to EfficientNet (head-only ceiling test), then ResNet50.**
- EfficientNet: head-only, AdamW wd=1e-3, patience=7, CosineAnnealingLR
- Result will reveal whether ceiling is MobileNetV2-specific or dataset-wide
- If EfficientNet also ≤60%: collect 300+ samples/class before ResNet50
- If EfficientNet reaches 65%+: ResNet50 next with same head-only settings

---

## ⚠️ MANDATORY AFTER EVERY SECTION (no exceptions)
When you see a POST-COMMIT REMINDER, do ALL THREE immediately:
1. **Update CLAUDE.md** — move section Remaining → Completed, update Active Section
2. **Update cross_project_ml.md** — add SmartVision lesson block
3. **Update INTERVIEW_PREP_SMARTVISION.md** — add interview-relevant Q&As from this section only (skip if no new interview-worthy decisions were made)

---

## Progress Tracker

### Completed ✅
- [x] Section 0: .claude/ setup (settings.json, rule files)
- [x] Section 1: config.py, requirements*.txt, .gitignore, pyrightconfig.json, __init__.py, CLAUDE.md
- [x] Section 2: Dataset Acquisition (HuggingFace streaming + checkpoint/resume) — 2,500 images, 25 classes, data in data/processed/
- [x] Section 3: EDA (class distribution, image quality, chi-squared balance test, 7 figures)
- [x] Section 4: Preprocessing + Augmentation + YOLO Annotation Validation

### In Progress 🔄
- [ ] Section 5: CNN Training (Colab T4)
  - [x] VGG16: 59.5% — accepted (architecture limitation, overfitting Phase 2)
  - [ ] ResNet50: layer4.2-only Phase 2 fix applied (fc unfreeze bug fixed), retrain PENDING
  - [ ] MobileNetV2: not started
  - [ ] EfficientNetB0: not started

### Remaining 📋
- [ ] Section 6: YOLOv8 Detection
- [ ] Section 7: Model Comparison + MLflow + Drift Baseline
- [ ] Section 8: FastAPI + Redis
- [ ] Section 9: KS Drift + Prometheus
- [ ] Section 10: Streamlit Multi-Page App
- [ ] Section 11: Docker Compose + Grafana + Airflow Scaffolding
- [ ] Section 12: CI + Tests + HuggingFace Deployment

---

## Project Architecture (Locked)

### Two-Phase Pipeline
**Phase A (Google Colab T4):** Stream COCO → preprocess → train 4 CNNs (parameterized) → train YOLOv8
→ verify → upload weights to HuggingFace Hub → commit artifacts to git

**Phase B (Docker Compose):** FastAPI lifespan → download weights from HF Hub (if not in volume)
→ load models → warm-up → serve `/classify`, `/detect`, `/metrics`
→ Redis caches results → Prometheus scrapes → Grafana dashboards

Streamlit NEVER loads models. It calls FastAPI endpoints.

### Stack
Python 3.11 | PyTorch 2.3.0 | torchvision 0.18.0 | Ultralytics YOLOv8 8.2.0
FastAPI 0.111.0 | Redis 7 | Prometheus | Grafana
MLflow 2.14.1 | Streamlit 1.37.0 | Plotly 5.22.0
Docker Compose | pytest | GitHub Actions CI

### Data Facts
- Dataset: COCO 2017 subset, 25 classes, 2,500 images (100 per class)
- Splits: 70% train (1,750) / 15% val (375) / 15% test (375)
- CNN input: 224×224, ImageNet normalization
- YOLO input: 640×640

### Locked Decisions
- RANDOM_STATE = 42 everywhere
- PyTorch (not TensorFlow)
- Training in Google Colab (T4/A100)
- VGG16 batch = 16 (not 32) — memory budget
- HF Hub for model weights (not Git LFS)
- lifespan context manager (not @app.on_event)
- KS drift on confidence scores (not raw embeddings)
- Redis is optional — graceful degradation on failure
- Notebooks are .py with # %% markers (jupytext for Colab)

---

## 38 Critical Rules (Quick Reference)

| # | Rule |
|---|------|
| 1 | FAST_MODE = local var in notebooks, passed as param; config.FAST_MODE reads env var |
| 2 | NumpyEncoder: torch.Tensor → np.bool_ → np.integer → np.floating → np.ndarray |
| 3 | One model trainer per session turn; wait for confirmation |
| 4 | No model() calls in pages/; src/ never imports from api/ (use DI) |
| 5 | torch.save(model.state_dict(), path) — never full pickle |
| 6 | ENV PYTHONPATH=/app in Dockerfile from Section 1 |
| 7 | pyrightconfig.json with extraPaths=["."] from Section 1 |
| 8 | YOLO exist_ok=True in model.train() |
| 9 | ImageNet normalization for all 4 CNNs |
| 10 | Discrete Plotly colors for category charts |
| 11 | st.selectbox: .get(selected or "", default) |
| 12 | Page structure: result first → st.expander("Technical") at bottom |
| 13 | Bool after CSV: .astype(str).str.lower().map({"true":...}) |
| 14 | MLflow params cast on read: int(run["params.epochs"]) |
| 15 | mlflow.search_runs(experiment_ids=[...]) not experiment_names= |
| 16 | Section completion: 6-step git checklist |
| 17 | 6dp in JSON artifacts, 4dp display |
| 18 | Update CLAUDE.md + cross_project_ml.md after every section commit |
| 19 | /data/raw/ in .gitignore with leading slash |
| 20 | pages/ routing, not st.navigation() |
| 21 | KS test on confidence scores — NOT raw embeddings |
| 22 | Docker: depends_on: {condition: service_healthy} |
| 23 | Redis: cache miss → inference, never raise |
| 24 | COCO class IDs NOT sequential — COCO_ID_TO_CLASS_IDX everywhere |
| 25 | VGG16 batch = 16 not 32 |
| 26 | HF Spaces = lightweight Streamlit; full Docker = local demo |
| 27 | .py notebooks with # %% markers; never commit .ipynb |
| 28 | verify_coco_mapping() as first call in 01_data_acquisition.py |
| 29 | libgl1-mesa-glx + libglib2.0-0 in Dockerfile BEFORE pip install |
| 30 | /metrics returns CONTENT_TYPE_LATEST (text/plain) |
| 31 | asyncio.get_running_loop() not get_event_loop() |
| 32 | lifespan context manager not @app.on_event("startup") |
| 33 | Prometheus label name consistent: class_name everywhere |
| 34 | __init__.py in all src/ and api/ subdirectories |
| 35 | Grafana provisioning directory for auto-datasource |
| 36 | torch.load(..., weights_only=True) everywhere |
| 37 | HfApi().upload_file(); create_repo(exist_ok=True) before first upload |
| 38 | @st.cache_data for artifact loading; @st.cache_resource for models |
| 39 | torch.load(..., weights_only=True) everywhere — security + FutureWarning |
| 40 | __file__ undefined in Jupyter/Colab — wrap PROJECT_ROOT in try/except NameError |
| 41 | Create INTERVIEW_PREP_SMARTVISION.md at Section 0/1; update after every section with interview-relevant Q&As only |

---

## Notebook Convention
Notebooks are `.py` files with `# %%` cell markers.
Convert for Colab: `jupytext --to notebook notebooks/04_train_classifier.py`
Never commit `.ipynb` files (in .gitignore).
