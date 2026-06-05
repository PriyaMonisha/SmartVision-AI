# filename: api/prometheus_metrics.py
# purpose:  Prometheus metric definitions — module-level singletons.
#           prometheus_client requires metrics to be created once at import time.
#           Rule 30: /metrics endpoint returns CONTENT_TYPE_LATEST (text/plain).
#           Rule 33: label name "class_name" used consistently here and in Section 9 rules.

from prometheus_client import Counter, Histogram, Gauge

# Classification metrics
# 22 classes × 2 models = 44 max label combinations — acceptable cardinality.
classify_requests = Counter(
    "smartvision_classify_requests_total",
    "Total classification requests by top-1 prediction and model",
    ["class_name", "model_name"],
)
classify_latency = Histogram(
    "smartvision_classify_latency_seconds",
    "End-to-end classification latency (excluding cache hits)",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
)

# Detection metrics
detect_requests = Counter(
    "smartvision_detect_requests_total",
    "Total detection requests",
)
detect_latency = Histogram(
    "smartvision_detect_latency_seconds",
    "End-to-end detection latency (excluding cache hits)",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
)

# Unknown-prediction counter — fires when model returns a class not in config.CLASSES.
# High rate indicates corrupted weights or wrong input preprocessing pipeline.
# Rule 33: label name model_name consistent with classify_requests label.
unknown_predictions = Counter(
    "smartvision_unknown_predictions_total",
    "Predictions with class_name not in CLASSES — indicates model or preprocessing issue",
    ["model_name"],
)

# Cache metrics — used by Section 9 to measure cache efficiency
cache_hits = Counter(
    "smartvision_cache_hits_total",
    "Redis cache hits by endpoint",
    ["endpoint"],
)
cache_misses = Counter(
    "smartvision_cache_misses_total",
    "Redis cache misses by endpoint",
    ["endpoint"],
)

# HTTP error rate — counts 4xx/5xx responses by status code and endpoint.
# Enables Grafana alerting on error spikes (e.g., >5% 500s triggers page).
http_errors = Counter(
    "smartvision_http_errors_total",
    "HTTP error responses by status code and endpoint",
    ["status_code", "endpoint"],
)

# Models-loaded gauge — 0 during startup, 1 when all expected models are ready.
# Alert rule: smartvision_models_loaded == 0 for > 3 minutes → startup failure.
models_loaded = Gauge(
    "smartvision_models_loaded",
    "1 when all models loaded successfully, 0 during startup or on load failure",
)
