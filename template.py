import os

BASE = os.path.dirname(os.path.abspath(__file__))

dirs = [
    f"{BASE}/data/raw",
    f"{BASE}/data/processed",
    f"{BASE}/data/external",
    f"{BASE}/experiments",                        
    f"{BASE}/src/data",
    f"{BASE}/src/features",
    f"{BASE}/src/models",
    f"{BASE}/src/training",
    f"{BASE}/src/explainability",
    f"{BASE}/src/serving",
    f"{BASE}/src/utils",
    f"{BASE}/configs",
    f"{BASE}/reports",
    f"{BASE}/monitoring/grafana",
    f"{BASE}/.github/workflows",                 
]

files = [
    # Data
    f"{BASE}/src/data/download.py",
    f"{BASE}/src/data/validate.py",
    # Features
    f"{BASE}/src/features/engineer.py",
    f"{BASE}/src/features/selector.py",
    # Models
    f"{BASE}/src/models/xgb_model.py",
    # Training
    f"{BASE}/src/training/train.py",
    f"{BASE}/src/training/evaluate.py",
    # Explainability
    f"{BASE}/src/explainability/shap_explainer.py",
    f"{BASE}/src/explainability/report_generator.py",
    # Serving
    f"{BASE}/src/serving/app.py",
    f"{BASE}/src/serving/schemas.py",
    f"{BASE}/src/serving/metrics.py",
    f"{BASE}/src/serving/middleware.py",
    # Utils
    f"{BASE}/src/utils/config.py",
    f"{BASE}/src/utils/logger.py",
    # Config
    f"{BASE}/configs/config.yaml",
    # MLOps
    f"{BASE}/dvc.yaml",
    f"{BASE}/dvc.lock",
    f"{BASE}/docker-compose.yml",
    f"{BASE}/Dockerfile",
    f"{BASE}/requirements.txt",
    # Monitoring
    f"{BASE}/monitoring/prometheus.yml",
    f"{BASE}/monitoring/grafana/dashboard.json",
    # CI/CD
    f"{BASE}/.github/workflows/ci.yml",
    f"{BASE}/.github/workflows/cd.yml",

]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"{d}")

for f in files:
    if not os.path.exists(f):
        open(f, 'w').close()
        print(f"{f}")

print("\nProject structure created!")