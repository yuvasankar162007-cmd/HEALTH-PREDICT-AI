"""
Explainable AI (SHAP) API Endpoints for HealthPredict.
Provides transparent feature attributions, base values, and clinical narratives.
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any
from pydantic import BaseModel

from backend.schemas import HealthInput, ShapExplanationResponse
from backend.services.xai_service import xai_service

router = APIRouter(prefix="/api/explain", tags=["Explainable AI (SHAP)"])


class ExplainRequest(BaseModel):
    disease: str = "diabetes"
    input_data: HealthInput


@router.post("/shap", response_model=ShapExplanationResponse)
def explain_prediction(payload: ExplainRequest):
    """
    Computes local SHAP attributions and clinical narrative for the given disease and patient data.
    """
    try:
        user_dict = payload.input_data.model_dump()
        result = xai_service.explain_disease(payload.disease, user_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability computation error: {str(e)}")

