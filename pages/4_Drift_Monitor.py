# filename: pages/4_Drift_Monitor.py
# purpose:  Per-class KS drift status via GET /drift/status.
#           Manual Refresh button (no auto-polling). st.session_state persists last fetch.
#           Alerting classes sorted first (numeric _sort_key, not string sort).

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from streamlit_app import api_client
from streamlit_app.plotting import drift_gauge

st.set_page_config(page_title="Drift Monitor — SmartVision AI", layout="wide")
st.title("KS Drift Monitor")
api_client.demo_banner()
st.caption(
    "Monitors per-class confidence score distribution vs the MobileNet val-split baseline "
    "(n=30/class). Alert fires when KS stat > 0.10 AND p-value < 0.05 (double-gate)."
)
st.markdown(
    "Click **Refresh** to fetch current drift status from FastAPI. "
    "Classes need at least 100 live inferences before KS test runs."
)

# ── Refresh button — no auto-polling (prevents Streamlit re-run loops) ─────────
if "drift_status" not in st.session_state:
    st.info("Click **Refresh** to load current drift status.")

if st.button("Refresh", type="primary"):
    try:
        st.session_state["drift_status"] = api_client.get_drift_status()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

# ── Render cached status (persists across reruns from sidebar, sliders, etc.) ─
if "drift_status" not in st.session_state:
    st.stop()

status  = st.session_state["drift_status"]
summary = status["summary"]

st.divider()

# ── Summary metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Classes with data",  summary["classes_with_data"])
c2.metric("Classes tested",     summary["classes_tested"])
c3.metric("Classes alerting",   summary["classes_alerting"],
          delta=f"{summary['classes_alerting']} alert(s)" if summary["classes_alerting"] else None,
          delta_color="inverse")
c4.metric("Baseline model",     summary["baseline_model"].upper()
          + f" ({summary['baseline_split']} split)")

# ── Gauge ─────────────────────────────────────────────────────────────────────
st.plotly_chart(drift_gauge(summary), use_container_width=False)

# ── Per-class table — alerting classes first via numeric sort key ─────────────
ALERT_PRIORITY = {"ALERT": 2, "OK": 1}  # Waiting = 0 (default)

rows = []
for cls, d in status["classes"].items():
    min_req = d.get("min_samples_required", 100)
    if d["tested"]:
        alert_str = "ALERT" if d["is_alert"] else "OK"
    else:
        alert_str = f"Waiting ({d['buffer_size']}/{min_req})"

    rows.append({
        "Class":       cls,
        "Buffer":      d["buffer_size"],
        "KS stat":     f"{d['ks_stat']:.4f}"  if d.get("ks_stat")  is not None else "—",
        "p-value":     f"{d['p_value']:.4f}"  if d.get("p_value")  is not None else "—",
        "Alert":       alert_str,
        "_sort_key":   ALERT_PRIORITY.get(alert_str.split()[0], 0),
    })

df = (
    pd.DataFrame(rows)
    .sort_values("_sort_key", ascending=False)
    .drop(columns=["_sort_key"])
    .reset_index(drop=True)
)

st.dataframe(df, use_container_width=True, hide_index=True)
st.caption("Alert rule: KS stat > 0.10 AND p-value < 0.05  |  Baseline: MobileNet val split, n=30/class")
