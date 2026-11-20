"""
Counterfactual Simulation API Endpoints for HealthPredict.
Enables dynamic 'What-If' lifestyle modification assessments.
"""

from fastapi import APIRouter, HTTPException, Query
from backend.schemas import WhatIfInterventionRequest, WhatIfInterventionResponse
from backend.services.simulator_service import simulator_service

router = APIRouter(prefix="/api/simulate", tags=["What-If Counterfactual Simulator"])


@router.post("/what-if", response_model=WhatIfInterventionResponse)
def simulate_what_if_lifestyle(
    payload: WhatIfInterventionRequest,
    low_thresh: float = Query(0.30, ge=0.05, le=0.50),
    high_thresh: float = Query(0.60, ge=0.51, le=0.90)
):
    """
    Simulates the quantitative risk impact of hypothetical lifestyle and biometric changes.
    """
    try:
        base_dict = payload.base_data.model_dump()
        result = simulator_service.simulate_intervention(
            base_dict=base_dict,
            weight_loss_kg=payload.weight_loss_kg,
            sbp_reduction=payload.sbp_reduction,
            glucose_reduction=payload.glucose_reduction,
            chol_reduction=payload.chol_reduction,
            quit_smoking=payload.quit_smoking,
            boost_activity=payload.boost_activity,
            low_thresh=low_thresh,
            high_thresh=high_thresh
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")

