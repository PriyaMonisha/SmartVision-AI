# filename: api/prometheus_metrics.py
# purpose:  Prometheus metric definitions — module-level singletons.
#           prometheus_client requires metrics to be created once at import time.
#           Rule 30: /metrics endpoint returns CONTENT_TYPE_LATEST (text/plain).
#           Rule 33: label name "class_name" used consistently here and in Section 9 rules.

from prometheus_client import Counter, Histogram

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
