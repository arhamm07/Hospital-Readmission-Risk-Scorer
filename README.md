# Hospital Readmission Risk Scorer

Predict 30-day readmission risk for diabetic patients. XGBoost + SHAP + calibrated probabilities + production MLOps.

---

## Quick Start

```bash
git clone <repo-url>
cd hospital-readmission-risk
pip install -r requirements.txt

dvc repro

docker-compose up -d
```

**Dashboards:**
- API docs → http://localhost:8000/docs
- MLflow → http://localhost:5000
- Grafana → http://localhost:3000 (admin/admin)

---

## What It Does

Predict which diabetic patients will be readmitted within 30 days. Enable proactive clinical intervention. Target: AUC-ROC >= 0.70, ECE <= 0.05.

---

## Architecture

```
Raw Data (UCI Diabetes 130-US)
         ↓
[DVC Pipeline]
  data_download → data_validate → feature_engineering
       → feature_selection → train → evaluate
         ↓
[MLflow Registry]
  XGBoost + Isotonic Calibration → Production stage
         ↓
[FastAPI Server]
  POST /predict-risk      → risk_score + SHAP + risk_tier
  POST /predict-risk/batch
  GET  /health
  GET  /model-info
  GET  /metrics
         ↓
[Prometheus + Grafana]
  Latency · Drift · Risk tiers · Errors
```

---

## Model Performance

| Metric | Target | Actual |
|--------|--------|--------|
| AUC-ROC | >= 0.70 | ~0.68 |
| AUC-PR | >= 0.30 | ~0.21 |
| ECE | <= 0.05 | ~0.005 |
| KS Statistic | - | ~0.27 |
| PPV (top 20%) | - | ~0.24 |
| NNT | - | ~4.9 |

---

## API Usage

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
}

resp = requests.post(
    "http://localhost:8000/predict-risk",
    json=patient,
    params={"patient_id": "P001"}
)
print(resp.json())
```

**Response:**
```json
{
  "patient_id": "P001",
  "risk_score": 0.4213,
  "risk_tier": "High",
  "top_factors": [
    {"feature": "n_active_medications", "shap_impact": 0.082},
    {"feature": "any_insulin", "shap_impact": 0.071}
  ]
}
```

---

## Commands

```bash
dvc dag                 # visualize pipeline
dvc repro               # run full pipeline
dvc metrics show       # current metrics
dvc metrics diff       # compare vs last run

docker-compose up -d   # start all services
docker-compose logs -f api
docker-compose down

pytest tests/ -v --cov=src
```

---

## Project Structure

```
hospital-readmission-risk/
├── configs/config.yaml          # all hyperparameters
├── data/
│   ├── raw/                    # DVC-tracked
│   ├── processed/              # DVC-tracked
│   └── external/
├── experiments/                 # Jupyter notebook
├── models/                     # DVC-tracked
├── reports/                    # plots + metrics
├── src/
│   ├── data/                   # download + validate
│   ├── features/              # engineer + select
│   ├── models/                # XGBoost wrapper
│   ├── training/              # train + evaluate
│   ├── explainability/        # SHAP
│   ├── serving/                # FastAPI
│   └── utils/                  # config + logger
├── monitoring/                 # Prometheus + Grafana
├── dvc.yaml
├── docker-compose.yml
└── Dockerfile
```

---

## Key Features

- **Feature Engineering:** 50+ features (demographics, labs, medications, diagnoses, interactions)
- **Feature Selection:** 3-method consensus (MI + LASSO + RF)
- **Model:** XGBoost + Optuna HPO (50 trials) + Isotonic calibration
- **Explainability:** SHAP for local + global explanations
- **Risk Tiers:** Low (< 0.25) / Medium (0.25-0.50) / High (0.50-0.75) / Critical (> 0.75)
- **Data Versioning:** DVC
- **Experiment Tracking:** MLflow
- **Monitoring:** Prometheus + Grafana

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Model | XGBoost |
| Calibration | Isotonic Regression |
| HPO | Optuna |
| Explainability | SHAP |
| Data Versioning | DVC |
| Experiment Tracking | MLflow |
| API | FastAPI |
| Monitoring | Prometheus + Grafana |
| Container | Docker |

---

## Environment Variables

```bash
MLFLOW_TRACKING_URI=http://localhost:5000
MODEL_URI=models:/hospital-readmission-risk/Production
```

---

## License

MIT