# filename: pages/2_Detect.py
# purpose:  Upload an image → POST /detect → annotated image with bboxes + detection table.
#           No model calls. EXIF rotation corrected on both display and drawing (plan C3).
#           Bbox coords from API are absolute pixels — drawn directly with PIL.

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config
from streamlit_app import api_client
from streamlit_app.plotting import class_color

st.set_page_config(page_title="Detect — SmartVision AI", layout="wide")
st.title("Object Detection")
st.caption("YOLOv8n via FastAPI /detect. Bounding boxes drawn on EXIF-corrected image.")
api_client.demo_banner()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(config.YOLO_CONF_THRESHOLD),
        step=0.05,
        help="Lower = more detections (including low-confidence ones).",
    )

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    help=f"Max {config.MAX_IMAGE_SIZE_MB} MB.",
)

if uploaded_file is None:
    st.info("Upload an image to run object detection.")
    st.stop()

# Read ONCE into bytes (plan C2)
img_bytes = uploaded_file.read()

if len(img_bytes) > config.MAX_IMAGE_SIZE_MB * 1024 ** 2:
    st.error(f"Image too large (max {config.MAX_IMAGE_SIZE_MB} MB). Please resize and retry.")
    st.stop()

# ── EXIF-corrected PIL image for display and drawing (plan C3) ────────────────
def _open_corrected(raw: bytes) -> Image.Image:
    """Open image and apply EXIF orientation — fixes rotated phone photos."""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return ImageOps.exif_transpose(img)

pil_image = _open_corrected(img_bytes)

# ── Inference ─────────────────────────────────────────────────────────────────
with st.spinner("Running YOLOv8n detection…"):
    try:
        result = api_client.detect(img_bytes, uploaded_file.name, conf_threshold)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

detections  = result.get("detections", [])
inf_time    = result.get("inference_time_ms", 0.0)
cached      = result.get("cached", False)

# ── Draw bboxes on EXIF-corrected image ───────────────────────────────────────
annotated = pil_image.copy()
draw      = ImageDraw.Draw(annotated)

for det in detections:
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    colour = class_color(det["class_name"])
    draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)
    label = f"{det['class_name']} {det['confidence']:.0%}"
    draw.rectangle([x1, y1 - 16, x1 + len(label) * 7 + 4, y1], fill=colour)
    draw.text((x1 + 4, y1 - 15), label, fill="white")

# ── Result-first layout (Rule 12) ─────────────────────────────────────────────
st.image(annotated, caption=f"{uploaded_file.name} — {len(detections)} detection(s)")

badge_col, time_col, det_col = st.columns(3)
if cached:
    badge_col.success("Cache hit")
else:
    badge_col.info("Fresh inference")
time_col.metric("Inference time", f"{inf_time:.1f} ms")
det_col.metric("Detections", len(detections))

if not detections:
    st.info(
        "No objects detected above the confidence threshold. "
        "Try lowering the slider in the sidebar."
    )
else:
    rows = [
        {
            "Class":      d["class_name"],
            "Confidence": f"{d['confidence']:.1%}",
            "x1":         int(d["x1"]),
            "y1":         int(d["y1"]),
            "x2":         int(d["x2"]),
            "y2":         int(d["y2"]),
        }
        for d in sorted(detections, key=lambda x: x["confidence"], reverse=True)
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Technical details expander (Rule 12) ──────────────────────────────────────
with st.expander("Technical details"):
    st.json(result)
    st.caption(
        f"Confidence threshold: {conf_threshold} | "
        f"Inference: {inf_time:.1f} ms | "
        f"Cached: {cached} | "
        f"Image size after EXIF correction: {pil_image.size}"
    )
