# filename: streamlit_app/plotting.py
# purpose:  Plotly chart builders for all Streamlit pages.
#           Discrete colour palette (Rule 10). No model calls, no API calls.

from __future__ import annotations

import sys
from pathlib import Path

import plotly.colors as pc
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ── 24-colour palette — covers all 22 COCO classes without forced collisions ──
_PALETTE: list[str] = (
    pc.qualitative.Plotly        # 10 colours
    + pc.qualitative.Pastel      # 10 colours
    + pc.qualitative.Dark24[:4]  # 4 more → 24 total
)


def class_color(class_name: str) -> str:
    """Return a stable colour for a class name.

    Known COCO classes (config.CLASSES) use their fixed index for deterministic,
    consistent colouring. Unknown YOLO class names fall back to hash-based assignment.
    """
    try:
        idx = config.CLASSES.index(class_name)
    except ValueError:
        idx = hash(class_name)
    return _PALETTE[idx % len(_PALETTE)]


# ── Accuracy bar chart ────────────────────────────────────────────────────────

def accuracy_bar(model_metrics: dict) -> go.Figure:
    """Horizontal bar chart of test accuracy for all 4 CNN classifiers."""
    models_data = model_metrics.get("models", {})
    champion = model_metrics.get("champion_classifier", "resnet50")

    names, accs, colours = [], [], []
    for name, d in models_data.items():
        acc = d.get("test_accuracy")
        if acc is None:
            continue
        names.append(name.upper())
        accs.append(acc)
        colours.append("#EF553B" if name == champion else "#636EFA")

    fig = go.Figure(go.Bar(
        x=accs,
        y=names,
        orientation="h",
        marker_color=colours,
        text=[f"{a:.1%}" for a in accs],
        textposition="outside",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Classification Test Accuracy",
        xaxis=dict(title="Test Accuracy", tickformat=".0%", range=[0, max(accs) * 1.15]),
        yaxis=dict(title=""),
        height=300,
        margin=dict(l=20, r=60, t=50, b=40),
        showlegend=False,
    )
    return fig


# ── Speed vs accuracy scatter ─────────────────────────────────────────────────

def speed_accuracy_scatter(model_metrics: dict) -> go.Figure:
    """Bubble scatter: x=latency, y=accuracy, bubble_size ∝ model_size_mb (normalised)."""
    models_data = model_metrics.get("models", {})

    names, x_ms, y_acc, raw_sizes, colours = [], [], [], [], []
    for name, d in models_data.items():
        acc = d.get("test_accuracy")
        ms  = d.get("cpu_inference_ms")
        mb  = d.get("model_size_mb")
        if acc is None or ms is None or mb is None:
            continue
        names.append(name.upper())
        x_ms.append(ms)
        y_acc.append(acc)
        raw_sizes.append(mb)
        colours.append(class_color(name))

    # Normalise bubble sizes to 20-60px range
    s_min, s_max = min(raw_sizes), max(raw_sizes)
    denom = max(s_max - s_min, 1.0)
    sizes_norm = [20 + 40 * (s - s_min) / denom for s in raw_sizes]

    fig = go.Figure(go.Scatter(
        x=x_ms,
        y=y_acc,
        mode="markers+text",
        marker=dict(size=sizes_norm, color=colours, opacity=0.85, line=dict(width=1, color="white")),
        text=names,
        textposition="top center",
        customdata=list(zip(raw_sizes, [f"{s:.1f} MB" for s in raw_sizes])),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Latency: %{x:.1f} ms<br>"
            "Accuracy: %{y:.1%}<br>"
            "Size: %{customdata[1]}<extra></extra>"
        ),
    ))
    fig.add_annotation(
        text="Bubble size = model file size (MB)",
        xref="paper", yref="paper",
        x=0.01, y=0.02, showarrow=False,
        font=dict(size=10, color="gray"),
    )
    fig.update_layout(
        title="Speed vs Accuracy (CPU inference, architecture benchmark)",
        xaxis=dict(title="CPU Inference Time (ms)"),
        yaxis=dict(title="Test Accuracy", tickformat=".0%"),
        height=400,
        margin=dict(l=20, r=20, t=50, b=40),
    )
    return fig


# ── Drift gauge ───────────────────────────────────────────────────────────────

def drift_gauge(summary: dict) -> go.Figure:
    """go.Indicator gauge: classes alerting vs total."""
    alerting = summary.get("classes_alerting", 0)
    total    = summary.get("total_classes", config.NUM_CLASSES)

    pct = alerting / max(total, 1)
    bar_colour = "green" if pct < 0.10 else ("orange" if pct < 0.30 else "crimson")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=alerting,
        title={"text": "Classes Alerting"},
        delta={"reference": 0, "increasing": {"color": "crimson"}},
        gauge={
            "axis":  {"range": [0, total], "tickwidth": 1},
            "bar":   {"color": bar_colour},
            "steps": [
                {"range": [0,             total * 0.10], "color": "lightgreen"},
                {"range": [total * 0.10,  total * 0.30], "color": "lightyellow"},
                {"range": [total * 0.30,  total],        "color": "lightsalmon"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 3},
                "thickness": 0.75,
                "value": total * 0.10,
            },
        },
    ))
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=50, b=20))
    return fig
