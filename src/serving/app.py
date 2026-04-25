import json 
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.serving.schemas import (
    PatientFeatures,
    BatchPredictRequest,
    BatchPredictResponse,
    RiskPrediction,
    SHAPFactor,
    HealthResponse,
    ModelInfoResponse,
)

from src.serving.metrics import (
    PREDICTION_COUNTER,
    MODEL_CONFIDENCE,
    RISK_TIER_COUNTER,
    BATCH_SIZE,
    MODEL_LOADED,
)

from src.serving.middleware import RequestLoggingMiddleware
from src.features.selector import FeatureSelector
from src.explainability.shap_explainer import SHAPExplainer


logger = get_logger(__name__, log_file= 'logs/serving.log')


class ModelState:
    cfg: None
    calibrated_model: None
    base_model: None
    shap_explainer: None
    feature_names: list[str] = []
    risk_thresholds: dict = {}
    model_metrics: dict = {}
    model_version: str = '1.0.0'
    is_ready: bool = False
    
state = ModelState()


def load_all_artifacts():
    """Loads model, features, SHAP explainer, thresholds at startup."""
    try:
        cfg = load_config()
        state.cfg = cfg
        proc = Path(cfg.data.processed_dir)
        
        
        state.feature_names = FeatureSelector.load(
            str(proc / 'selected_features.json')
        )
        
        logger.info(f"Selected features loaded: {state.feature_names}")
        
        cal_path = Path('models/calibrated_model.pkl')
        with open(cal_path, 'rb') as f:
            state.calibrated_model = pickle.load(f)
        logger.info(f"Calibrated model loaded from {cal_path}")
        
        state.base_model = xgb.XGBClassifier()
        state.base_model.load_model('models/xgb_base.ubj')
        logger.info(f'Base model loaded successfully')
        
        
        train_path = proc / 'train.parquet'
        if train_path.exists():
            train_df = pd.read_parquet(train_path)
            TARGET = cfg.data.target_binary_col
            feat_in_train = [f for f in state.feature_names if f in train_df.columns]
            X_bg = train_df[feat_in_train].values[:500]
            
            state.shap_explainer = SHAPExplainer(
                lambda x: state.base_model.predict_proba(x)[:, 1],
                feat_in_train,
                cfg,
            )
            
            state.shap_explainer.fit(X_bg)
            logger.info(f"SHAP Explainer initialized with background data from {train_path}")
            
        else:
            logger.warning(f"Training data not found at {train_path}. SHAP Explainer not initialized.")
            
            
        metrics_path = Path('models/risk_tier_thresholds.json')
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                state.risl_thresholds = json.load(f)
                
                
        state.is_ready = True
        MODEL_LOADED.set(1)
        logger.info("All artifacts loaded successfully. Model is ready to serve.")
        
        
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        MODEL_LOADED.set(0)
        raise 
    


def assign_risk_tier(score: float, thresholds: dict) -> str:
    p25 = thresholds.get("p25", 0.15)
    p50 = thresholds.get("p50", 0.25)
    p75 = thresholds.get("p75", 0.40)
    if score < p25:   return "Low"
    elif score < p50: return "Medium"
    elif score < p75: return "High"
    else:             return "Critical"
    
    
def patient_to_array(
    patient: PatientFeatures,
    feature_names: list[str]
) -> np.ndarray:
    """
    Converts a PatientFeatures Pydantic object into a numpy row
    ordered to match feature_names exactly.
    Missing features are filled with 0.
    """
    patient_dict = patient.model_dump()
    row = [float(patient_dict.get(f, 0.0)) for f in feature_names]
    return np.array(row, dtype=np.float32).reshape(1, -1)


def predict_single(
    patient: PatientFeatures,
    patient_id: str = "unknown",
) -> RiskPrediction:
    """Core inference logic for a single patient."""

    x = patient_to_array(patient, state.feature_names)

    # Score
    risk_score = float(
        state.calibrated_model.predict_proba(x)[0, 1]
    )

    # Risk tier
    risk_tier = assign_risk_tier(risk_score, state.risk_thresholds)

    # SHAP explanation
    top_factors: list[SHAPFactor] = []
    if state.shap_explainer:
        try:
            exp = state.shap_explainer.explain_local(
                x[0],
                patient_id=patient_id,
                save_plot=False,
            )
            top_factors = [SHAPFactor(**f) for f in exp["top_factors"]]
            base_rate   = exp["base_value"]
        except Exception as e:
            logger.warning(f"SHAP failed for {patient_id}: {e}")
            base_rate = 0.11    # fallback to dataset average
    else:
        base_rate = 0.11
        
    # Prometheus
    PREDICTION_COUNTER.labels(
        model_version=state.model_version,
        risk_tier=risk_tier,
    ).inc()
    MODEL_CONFIDENCE.observe(risk_score)
    RISK_TIER_COUNTER.labels(tier=risk_tier).inc()

    return RiskPrediction(
        patient_id    = patient_id,
        risk_score    = round(risk_score, 4),
        risk_tier     = risk_tier,
        base_rate     = round(base_rate, 4),
        top_factors   = top_factors,
        model_version = state.model_version,
    )
    
    
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artefacts on startup, cleanup on shutdown."""
    logger.info("Starting up — loading model artefacts ...")
    load_all_artifacts()
    yield
    logger.info("Shutting down ...")
    MODEL_LOADED.set(0)
    
    
app = FastAPI(
    title= 'Hospital Readmission Risk Scorer',
    description= 'Predicts 30 days readmission risk for discharged patients using XGBoost and SHAP explanations.',
    version= '1.0.0',
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from pathlib import Path
static_dir = Path(__file__).resolve().parents[2] / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), "static")



@app.get('/health', response_model= HealthResponse, tags= ['Ops'])
def health_check():
    return HealthResponse(
        status= 'ok' if state.is_ready else 'loading',
        model_loaded = state.is_ready,
        model_version = state.model_version,
        )
    
    
@app.get("/model-info", response_model=ModelInfoResponse, tags=["Ops"])
def model_info():
    return ModelInfoResponse(
        model_name    = "hospital-readmission-risk",
        version       = state.model_version,
        auc_roc       = state.model_metrics.get("test_auc_roc"),
        auc_pr        = state.model_metrics.get("test_auc_pr"),
        ece           = state.model_metrics.get("test_ece"),
        n_features    = len(state.feature_names),
        feature_names = state.feature_names,
    )

    
    
@app.post('/predict-risk', response_model= RiskPrediction, tags= ['Inference'])
def predict_risk(patient: PatientFeatures,
                 request: Request,
                 patient_id: str = 'unknown'):
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        result = predict_single(patient, patient_id)
        return result
    except Exception as e:
        logger.error(f"Prediction error for {patient_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/predict-risk-debug', tags=['Debug'])
def predict_risk_debug(patient: dict, patient_id: str = 'unknown'):
    """Debug endpoint that skips validation."""
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        x = np.array([[float(patient.get(f, 0.0)) for f in state.feature_names]], dtype=np.float32)

        risk_score = float(state.calibrated_model.predict_proba(x)[0, 1])
        risk_tier = assign_risk_tier(risk_score, state.risk_thresholds)

        return {"patient_id": patient_id, "risk_score": risk_score, "risk_tier": risk_tier}
    except Exception as e:
        logger.error(f"Debug prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/predict-risk/batch",response_model=BatchPredictResponse,tags=["Inference"],)
def predict_risk_batch(payload: BatchPredictRequest):
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if len(payload.patients) == 0:
        raise HTTPException(status_code=400, detail="Empty patient list")

    BATCH_SIZE.observe(len(payload.patients))

    predictions = []
    for i, patient in enumerate(payload.patients):
        try:
            pred = predict_single(patient, patient_id=f"batch_{i}")
            predictions.append(pred)
        except Exception as e:
            logger.error(f"Batch error at index {i}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error at patient index {i}: {e}"
            )

    return BatchPredictResponse(
        predictions = predictions,
        n_patients  = len(predictions),
    )

@app.get("/", tags=["UI"], response_class=HTMLResponse)
def root():
    """Serve web UI."""
    from pathlib import Path
    template_path = Path(__file__).resolve().parents[2] / "templates" / "index.html"
    return HTMLResponse(content=template_path.read_text())


@app.get("/metrics", tags=["Ops"], include_in_schema=False)
def prometheus_metrics():
    """Prometheus scrape endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )