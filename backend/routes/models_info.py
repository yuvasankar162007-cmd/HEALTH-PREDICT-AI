"""
Model Information and Benchmark API Endpoints for HealthPredict.
Returns model evaluation matrices, feature dictionaries, and clinical documentation.
"""

from fastapi import APIRouter
from backend.services.ml_service import ml_service

router = APIRouter(prefix="/api/models", tags=["Model Hub & Benchmarks"])


@router.get("/benchmarks")
def get_benchmarks():
    """Returns benchmark comparison metrics across all 5 trained models per disease."""
    metrics = ml_service.model_metrics
    if not metrics:
        return {}
    
    # Structure grouped by disease for clean client consumption
    grouped = {}
    if "all_benchmarks" in metrics:
        for item in metrics["all_benchmarks"]:
            d = item.get("disease")
            if d not in grouped:
                grouped[d] = []
            grouped[d].append(item)
            
    return {
        "all_benchmarks": metrics.get("all_benchmarks", []),
        "best_models": metrics.get("best_models", {}),
        "grouped": grouped,
        "diabetes": grouped.get("diabetes", []),
        "heart": grouped.get("heart", []),
        "hypertension": grouped.get("hypertension", [])
    }


@router.get("/metadata")
def get_metadata():
    """Returns biomedical feature definitions, validation bounds, and clinical rationales."""
    return ml_service.feature_metadata
