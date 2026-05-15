# Claude Context — SmartVision AI
*Paste this at the start of every new Claude conversation for this project.*

---

## Who I Am

Data science student (GUVI capstone program) building production-grade ML systems.
Comfortable with Python, pandas, PyTorch basics, Streamlit, and Docker.
Work on **Windows 11** with Git Bash and PowerShell. Python 3.11.
Train models on **Google Colab** (T4/A100 GPU). Deploy to HuggingFace Spaces.

---

## How We Work Together (Non-Negotiable)

1. **Plan before coding.** State plan, call out assumptions, WAIT for "go" before writing code.
2. **ONE model per turn.** VGG16 → confirm → ResNet50 → confirm → MobileNet → confirm → EfficientNet.
3. **Present tradeoffs** when multiple valid approaches exist. Never silently pick one.
4. **Complete files only.** No `# ... rest of file unchanged`.
5. **Verify before referencing.** Read a file before assuming its contents.
6. **Never invent APIs.** If unsure a function exists, say so.

---

## Tech Stack

| Layer | Tool | Version |
|---|---|---|
| Language | Python | 3.11 |
| Deep Learning | PyTorch + torchvision | 2.3.0 / 0.18.0 |
| Object Detection | Ultralytics YOLOv8 | 8.2.0 |
| Dataset | HuggingFace datasets (COCO) | 2.20.0 |
| Model serving | FastAPI + uvicorn | 0.111.0 / 0.30.1 |
| Cache | Redis | 7 (Docker) |
| Monitoring | Prometheus + Grafana | latest |
| Experiment tracking | MLflow | 2.14.1 |
| UI | Streamlit | 1.37.0 |
| Charts | Plotly | 5.22.0 |
| Testing | pytest + httpx | 8.3.2 / 0.27.0 |
| Containers | Docker Compose | local stack |

---

## Architecture (LOCKED)

### Two-Phase Pipeline
- **Phase A (Colab):** stream COCO → preprocess → train 4 CNNs (parameterized, 1 per turn) + YOLOv8 → upload weights to HF Hub → commit artifacts to git
- **Phase B (Docker):** FastAPI lifespan → download weights if missing → load → warm-up → serve

**Streamlit NEVER loads models. Always calls FastAPI `/classify` and `/detect`.**

### 25 Classes (COCO subset)
Vehicles (6): car, truck, bus, motorcycle, bicycle, airplane
People (1): person
Outdoor (3): traffic light, stop sign, bench
Animals (6): dog, cat, horse, bird, cow, elephant
Kitchen/Food (5): bottle, cup, bowl, pizza, cake
Furniture (4): chair, couch, bed, potted plant

---

## Critical Rules to Always Follow

1. `FAST_MODE = True` at top of every training notebook as LOCAL var — pass to functions as param
2. NumpyEncoder: `torch.Tensor → np.bool_ → np.integer → np.floating → np.ndarray`
3. `torch.load(..., weights_only=True)` always
4. `lifespan` context manager (not `@app.on_event`)
5. `asyncio.get_running_loop()` (not `get_event_loop()`)
6. `HfApi().upload_file()` — `hf_hub_upload()` does not exist
7. KS drift uses confidence scores (1-dim) — NOT raw embeddings
8. Redis: cache miss → inference, never raise on `RedisError`
9. `COCO_ID_TO_CLASS_IDX` everywhere — COCO IDs are NOT sequential 0-24
10. `libgl1-mesa-glx + libglib2.0-0` in Dockerfile BEFORE pip install
11. `/metrics` returns `CONTENT_TYPE_LATEST` (text/plain, not JSON)
12. `depends_on: {condition: service_healthy}` in Docker Compose
13. `@st.cache_data` for artifacts; `@st.cache_resource` for models
14. Prometheus label: `class_name` (consistent in metrics.py, drift.py, and rules YAML)

---

## Section Completion Checklist

- [ ] All files saved
- [ ] Tests passed
- [ ] Artifacts present
- [ ] `git add <files>`
- [ ] `git commit -m "section-X: description"`
- [ ] `git log --oneline` — confirm
- [ ] `git status` — clean
- [ ] CLAUDE.md updated
- [ ] cross_project_ml.md updated

---

## Project Numbers

| Item | Value |
|---|---|
| Dataset | COCO 2017, 25 classes, 2,500 images |
| Images per class | 100 (10 in FAST_MODE) |
| Split | 70/15/15 |
| CNN input | 224×224, ImageNet normalization |
| YOLO input | 640×640 |
| VGG16 batch | 16 (not 32 — memory) |
| Total FastAPI memory | ~690MB → Docker mem_limit: 1.5g |
| KS drift threshold | 0.10 |
| YOLO conf threshold | 0.50 |
| HF Spaces | Streamlit SDK, CPU-only, EfficientNet + YOLOv8 only |
| Local Docker | Full stack (5 services + Airflow optional) |

---

## LESSONS — SmartVision AI

*(Updated after each section commit)*

### Section 0-1: Foundation
- Pydantic Settings handles .env reading cleanly — no raw os.getenv scattered everywhere
- requirements-hf.txt: CPU-only PyTorch needs `--index-url` per package, not just in filename
- 38 rules consolidated into CLAUDE.md quick-reference table — no excuse for missing them
