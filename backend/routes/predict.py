"""
Prediction API Endpoints for HealthPredict.
Handles single and multi-disease screening inferences and test persona loading.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Dict, Any, Optional

from backend.schemas import (
    HealthInput,
    ComprehensivePredictionResponse,
    SingleDiseasePrediction
)
from backend.services.ml_service import ml_service
from backend.database import save_assessment

router = APIRouter(prefix="/api/predict", tags=["Prediction Engine"])


@router.post("/all", response_model=ComprehensivePredictionResponse)
def predict_all_diseases(
    payload: HealthInput,
    save_to_history: bool = Query(False, description="Whether to persist this screening assessment to database"),
    low_thresh: float = Query(0.30, ge=0.05, le=0.50, description="Cutoff for low-risk band"),
    high_thresh: float = Query(0.60, ge=0.51, le=0.90, description="Cutoff for high-risk band")
):
    """
    Executes comprehensive disease risk screening across Diabetes, Coronary Heart Disease, and Hypertension.
    Includes automated BMI classification, composite lifestyle quality score, and personalized insights.
    """
    try:
        user_dict = payload.model_dump()
        result = ml_service.predict_all(user_dict, low_thresh, high_thresh)
        
        saved_id = None
        if save_to_history:
            record_data = {
                'patient_name': payload.patient_name or 'Anonymous User',
                'age': payload.age,
                'gender': payload.gender,
                'bmi': result['bmi_info']['bmi'],
                'blood_pressure_sys': payload.blood_pressure_sys,
                'blood_pressure_dia': payload.blood_pressure_dia,
                'glucose': payload.glucose,
                'cholesterol': payload.cholesterol,
                'lifestyle_score': result['lifestyle_score']['score'],
                'lifestyle_rating': result['lifestyle_score']['rating'],
                'diabetes_prob': result['diabetes']['probability'],
                'diabetes_cat': result['diabetes']['category'],
                'heart_prob': result['heart']['probability'],
                'heart_cat': result['heart']['category'],
                'hypertension_prob': result['hypertension']['probability'],
                'hypertension_cat': result['hypertension']['category'],
                'overall_risk_score': result['overall_risk_score'],
                'input_data': user_dict,
                'notes': payload.notes or ''
            }
            saved_id = save_assessment(record_data)
            
        result['saved_record_id'] = saved_id
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/diabetes", response_model=SingleDiseasePrediction)
def predict_diabetes_only(
    payload: HealthInput,
    low_thresh: float = Query(0.30, ge=0.05, le=0.50),
    high_thresh: float = Query(0.60, ge=0.51, le=0.90)
):
    """Runs isolated Type 2 Diabetes screening prediction."""
    try:
        return ml_service.predict_diabetes(payload.model_dump(), low_thresh, high_thresh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diabetes prediction error: {str(e)}")


@router.post("/heart", response_model=SingleDiseasePrediction)
def predict_heart_only(
    payload: HealthInput,
    low_thresh: float = Query(0.30, ge=0.05, le=0.50),
    high_thresh: float = Query(0.60, ge=0.51, le=0.90)
):
    """Runs isolated Coronary Heart Disease screening prediction."""
    try:
        return ml_service.predict_heart_disease(payload.model_dump(), low_thresh, high_thresh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Heart disease prediction error: {str(e)}")


@router.post("/hypertension", response_model=SingleDiseasePrediction)
def predict_hypertension_only(
    payload: HealthInput,
    low_thresh: float = Query(0.30, ge=0.05, le=0.50),
    high_thresh: float = Query(0.60, ge=0.51, le=0.90)
):
    """Runs isolated Hypertension screening prediction."""
    try:
        return ml_service.predict_hypertension(payload.model_dump(), low_thresh, high_thresh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hypertension prediction error: {str(e)}")


@router.get("/presets")
def get_presets():
    """Returns curated patient personas for demonstration and testing."""
    return ml_service.get_preset_personas()

