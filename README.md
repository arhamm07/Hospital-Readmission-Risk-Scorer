# 🏥 Hospital Readmission Risk Scorer

> **Project 1 of 30 — Traditional ML Engineering Mastery**
> 30-day readmission risk prediction with XGBoost, SHAP explanations,
> calibrated probabilities, and production MLOps.

---

## 🚀 Quick Start

```bash
# 1. Clone & install
git clone <repo-url>
cd readmission-risk
pip install -r requirements.txt

# 2. Run full pipeline
dvc repro

# 3. Start all services
docker-compose up -d

# 4. Open dashboards
#    API docs  → http://localhost:8000/docs
#    MLflow    → http://localhost:5000
#    Grafana   → http://localhost:3000  (admin/admin)
```

---

## 📐 Architecture

```
Raw Data (UCI)
     │
     ▼
[DVC Pipeline]
  data_download  →  data_validate  →  feature_engineering
       →  feature_selection  →  train  →  evaluate
     │
     ▼
[MLflow Registry]
  XGBoost + Isotonic Calibration
  Model version: Production
     │
     ▼
[FastAPI Server]
  POST /predict-risk
  POST /predict-risk/batch
  GET  /health
  GET  /model-info
  GET  /metrics  (Prometheus)
     │
     ▼
[Prometheus + Grafana]
  Latency · Drift · Risk tiers · Errors
```

---

## 📊 Model Performance

| Metric           | Value  |
|-----------------|--------|
| AUC-ROC          | ~0.71  |
| AUC-PR           | ~0.34  |
| KS Statistic     | ~0.32  |
| Gini             | ~0.42  |
| ECE (calibrated) | ~0.02  |
| PPV (top 20%)    | ~0.24  |
| NNT              | ~4.1   |

---

## 🔁 DVC Pipeline

```bash
dvc dag          # view pipeline DAG
dvc repro        # run full pipeline
dvc metrics show # show current metrics
dvc metrics diff # compare vs last commit
```

---

## 🧪 Run Tests

```bash
pytest tests/ -v --cov=src
```

---

## 🐳 Docker

```bash
docker-compose up -d          # start all services
docker-compose logs -f api    # follow API logs
docker-compose down           # stop all
```

---

## 🗂️ Project Structure

```
readmission-risk/
├── configs/config.yaml         ← all hyperparameters
├── data/
│   ├── raw/                    ← DVC-tracked
│   ├── processed/              ← DVC-tracked
│   └── external/
├── experiments/                ← Jupyter notebook
├── models/                     ← DVC-tracked
├── reports/                    ← plots + metrics
├── src/
│   ├── data/                   ← download + validate
│   ├── features/               ← engineer + select
│   ├── models/                 ← XGBoost wrapper
│   ├── training/               ← train + evaluate
│   ├── explainability/         ← SHAP
│   ├── serving/                ← FastAPI
│   └── utils/                  ← config + logger
├── monitoring/                 ← Prometheus + Grafana
├── .github/workflows/          ← CI + CD
├── dvc.yaml
├── docker-compose.yml
└── Dockerfile
```

---

## 📡 API Usage

```python
import requests

patient = {
    "age_numeric": 7,
    "gender": 0,
    "age_risk_bucket": "Elderly",
    "time_in_hospital": 5,
    "number_diagnoses": 9,
    "num_medications": 18,
    "num_lab_procedures": 45,
    "n_active_medications": 3,
    "any_insulin": 1,
    "total_prior_visits": 3,
    # ... other features
}

resp = requests.post(
    "http://localhost:8000/predict-risk",
    json=patient,
    params={"patient_id": "P001"}
)
print(resp.json())
# {
#   "patient_id": "P001",
#   "risk_score": 0.4213,
#   "risk_tier": "High",
#   "top_factors": [
#     {"feature": "n_active_medications", "shap_impact": 0.082, ...},
#     {"feature": "any_insulin", "shap_impact": 0.071, ...},
#   ]
# }