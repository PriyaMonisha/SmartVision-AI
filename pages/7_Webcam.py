# filename: pages/7_Webcam.py
# purpose:  Live webcam object detection using st.camera_input → POST /detect.
#           Continuous mode re-runs automatically for a live-detection feel.
#           Bounding boxes drawn with PIL (same palette as pages/2_Detect.py).

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config
from streamlit_app import api_client
from streamlit_app.plotting import class_color

st.set_page_config(page_title="Webcam Detection — SmartVision AI", layout="wide")
st.title("Live Webcam Detection")
st.caption("Captures a frame from your webcam and runs YOLOv8n detection via FastAPI /detect.")
api_client.demo_banner()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.10, max_value=0.90,
        value=float(config.YOLO_CONF_THRESHOLD), step=0.05,
    )
    continuous = st.toggle(
        "Continuous mode",
        value=False,
        help="Automatically re-captures and runs detection in a loop.",
    )
    st.divider()
    st.markdown("""
    **How it works**
    1. Click **Take photo** in the camera widget
    2. Detection runs automatically via FastAPI
    3. Enable **Continuous mode** for live feed
    """)

# ── Camera capture ────────────────────────────────────────────────────────────
camera_image = st.camera_input("Point camera at objects and click Take photo")

if camera_image is None:
    st.info("Allow camera access and click **Take photo** to start detection.")
    st.stop()

img_bytes = camera_image.read()

# ── EXIF correction ───────────────────────────────────────────────────────────
pil_img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

# ── Call /detect ──────────────────────────────────────────────────────────────
t_start = time.perf_counter()
try:
    result = api_client.detect(img_bytes, "webcam.jpg", conf_threshold)
except RuntimeError as e:
    st.error(f"API error: {e}")
    st.stop()

wall_ms = (time.perf_counter() - t_start) * 1000

# Filter by confidence threshold
detections = [
    d for d in result.get("detections", [])
    if d["confidence"] >= conf_threshold
]

# ── Metrics row ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Objects detected", len(detections))
col2.metric("Inference time", f"{result.get('inference_time_ms', 0):.0f} ms")
col3.metric("Wall time", f"{wall_ms:.0f} ms")
col4.metric("Cache hit", "Yes" if result.get("cached") else "No")

# ── Draw bounding boxes ───────────────────────────────────────────────────────
draw = ImageDraw.Draw(pil_img)
try:
    font = ImageFont.truetype("arial.ttf", max(14, pil_img.width // 60))
except OSError:
    font = ImageFont.load_default()

for det in detections:
    color = class_color(det["class_name"])
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    label = f"{det['class_name']} {det['confidence']:.0%}"
    bbox  = draw.textbbox((x1, y1 - 2), label, font=font)
    draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=color)
    draw.text((x1, y1 - 2), label, fill="white", font=font, anchor="lb")

# ── Display annotated image ───────────────────────────────────────────────────
st.image(pil_img, use_container_width=True, caption="YOLOv8n detections")

# ── Detection table ───────────────────────────────────────────────────────────
if detections:
    with st.expander("Detection details", expanded=False):
        rows = [
            {
                "Class": d["class_name"],
                "Confidence": f"{d['confidence']:.1%}",
                "x1": int(d["x1"]), "y1": int(d["y1"]),
                "x2": int(d["x2"]), "y2": int(d["y2"]),
            }
            for d in sorted(detections, key=lambda x: x["confidence"], reverse=True)
        ]
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.warning(
        f"No objects detected above {conf_threshold:.0%} confidence. "
        "Try lowering the threshold or moving the camera closer."
    )

# ── Continuous mode ───────────────────────────────────────────────────────────
if continuous:
    st.info("Continuous mode active — retaking photo in 1 second...")
    time.sleep(1)
    st.rerun()
