"""
Patient Assessment History API Endpoints for HealthPredict.
Provides SQLite CRUD operations and longitudinal telemetry tracking.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any

from backend.schemas import AssessmentRecordResponse, StatsSummaryResponse
from backend.database import (
    get_assessments,
    get_assessment_by_id,
    save_assessment,
    delete_assessment,
    get_stats_summary
)

router = APIRouter(prefix="/api/history", tags=["Patient Assessment History"])


@router.get("", response_model=List[AssessmentRecordResponse])
def list_assessments(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None, description="Search by patient name or notes"),
    risk_filter: Optional[str] = Query(None, description="Filter by risk category ('LOW RISK', 'MODERATE RISK', 'HIGH RISK')")
):
    """Retrieves paginated historical screening assessments."""
    try:
        return get_assessments(limit=limit, offset=offset, search=search, risk_filter=risk_filter)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database retrieval error: {str(e)}")


@router.get("/stats/summary", response_model=StatsSummaryResponse)
def get_history_stats():
    """Returns aggregated historical screening statistics and risk distribution."""
    try:
        return get_stats_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats calculation error: {str(e)}")


@router.get("/{assessment_id}", response_model=AssessmentRecordResponse)
def get_single_assessment(assessment_id: int):
    """Retrieves a specific assessment record by ID."""
    record = get_assessment_by_id(assessment_id)
    if not record:
        raise HTTPException(status_code=404, detail="Assessment record not found")
    return record


@router.delete("/{assessment_id}")
def delete_single_assessment(assessment_id: int):
    """Deletes an assessment record by ID."""
    success = delete_assessment(assessment_id)
    if not success:
        raise HTTPException(status_code=404, detail="Assessment record not found or could not be deleted")
    return {"message": "Assessment record deleted successfully", "id": assessment_id}

