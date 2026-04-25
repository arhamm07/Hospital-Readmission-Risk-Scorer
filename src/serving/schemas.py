from pydantic import BaseModel ,Field, field_validator
from typing import Any


class PatientFeatures(BaseModel):
    """
    Input schema for a single patient inference request.
    All fields map directly to engineered features.
    """
    # Demographic
    age_numeric:            float = Field(..., ge=0, le=9,  description="Age bucket index 0-9")
    gender:                 int   = Field(..., ge=-1, le=1, description="-1=unknown, 0=male, 1=female")
    age_risk_bucket:        str   = Field(..., description="Young / Middle / Elderly / Unknown")

    # Admission
    total_prior_visits:     float = Field(..., ge=0)
    high_utilizer:          int   = Field(..., ge=0, le=1)
    has_number_inpatient:   int   = Field(0, ge=0, le=1)
    has_number_emergency:   int   = Field(0, ge=0, le=1)
    has_number_outpatient:  int   = Field(0, ge=0, le=1)

    # Clinical
    time_in_hospital:       float = Field(..., ge=1, le=30)
    number_diagnoses:       float = Field(..., ge=0, le=20)
    num_medications:        float = Field(..., ge=0, le=100)
    num_lab_procedures:     float = Field(..., ge=0, le=200)
    num_procedures:         float = Field(..., ge=0, le=10)
    clinical_complexity:    float = Field(0.0, ge=0)
    days_per_diagnosis:     float = Field(0.0, ge=0)
    procedure_intensity:    float = Field(0.0, ge=0)
    lab_intensity:          float = Field(0.0, ge=0)

    # Lab
    a1c_tested:             int   = Field(0, ge=0, le=1)
    a1c_high:               int   = Field(0, ge=0, le=1)
    a1c_very_high:          int   = Field(0, ge=0, le=1)
    glucose_tested:         int   = Field(0, ge=0, le=1)
    glucose_high:           int   = Field(0, ge=0, le=1)
    both_labs_done:         int   = Field(0, ge=0, le=1)

    # Medications
    n_active_medications:   float = Field(0.0, ge=0)
    any_insulin:            int   = Field(0, ge=0, le=1)
    insulin_increased:      int   = Field(0, ge=0, le=1)
    insulin_decreased:      int   = Field(0, ge=0, le=1)
    any_med_change:         int   = Field(0, ge=0, le=1)
    on_diabetes_med:        int   = Field(0, ge=0, le=1)

    # Diagnosis categories
    diag_1_category:        str   = Field("Other")
    diag_2_category:        str   = Field("Other")
    diag_3_category:        str   = Field("Other")
    primary_diag_high_risk: int   = Field(0, ge=0, le=1)

    # Interactions
    age_x_ndiagnoses:       float = Field(0.0)
    n_meds_x_ndiagnoses:    float = Field(0.0)
    inpatient_x_emergency:  float = Field(0.0)
    elderly_high_complexity:int   = Field(0, ge=0, le=1)

    # Admission metadata
    admission_type_id:          float = Field(0.0)
    discharge_disposition_id:   float = Field(0.0)
    admission_source_id:        float = Field(0.0)

    class Config:
        extra = "allow"    # allow extra fields (future features)
        
        
class BatchPredictRequest(BaseModel):
    patients: list[PatientFeatures]
    
    

class SHAPFactor(BaseModel):
    feature: str
    value: float
    shap_impact: float
    direction: str
    
    
class RiskPrediction(BaseModel):
    patient_id:     str
    risk_score:     float   = Field(..., description="Calibrated probability [0,1]")
    risk_tier:      str     = Field(..., description="Low / Medium / High / Critical")
    base_rate:      float   = Field(..., description="Population average")
    top_factors:    list[SHAPFactor]
    model_version:  str
    
    
class BatchPredictResponse(BaseModel):
    predictions: list[RiskPrediction]
    n_patients:  int


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_version: str


class ModelInfoResponse(BaseModel):
    model_name:    str
    version:       str
    auc_roc:       float | None
    auc_pr:        float | None
    ece:           float | None
    n_features:    int
    feature_names: list[str]