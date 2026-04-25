from prometheus_client import Counter, Histogram, Gauge


PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_seconds",
    "Time taken for a single inference request",
    ["model_version", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

PREDICTION_COUNTER = Counter(
    "ml_predictions_total",
    "Total number of predictions served",
    ["model_version", "risk_tier"],
)

PREDICTION_ERRORS = Counter(
    "ml_prediction_errors_total",
    "Total number of prediction errors",
    ["error_type"],
)

FEATURE_DRIFT = Gauge(
    "ml_feature_drift_score",
    "PSI drift score per feature (updated periodically)",
    ["feature_name"],
)

MODEL_CONFIDENCE = Histogram(
    "ml_model_confidence",
    "Distribution of predicted risk scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

RISK_TIER_COUNTER = Counter(
    "ml_risk_tier_total",
    "Count of predictions per risk tier",
    ["tier"],
)

BATCH_SIZE = Histogram(
    "ml_batch_size",
    "Number of patients per batch request",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500],
)

MODEL_LOADED = Gauge(
    "ml_model_loaded",
    "1 if model is loaded and ready, 0 otherwise",
)