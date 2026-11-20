"""
Batch Processing API Endpoints for HealthPredict.
Handles CSV file uploads, multi-patient inference, and sample template retrieval.
"""

from fastapi import APIRouter, UploadFile, File, Query, HTTPException, Response
from backend.services.batch_service import batch_service

router = APIRouter(prefix="/api/batch", tags=["Batch Processing"])


@router.post("/upload")
async def upload_batch_csv(
    file: UploadFile = File(...),
    low_thresh: float = Query(0.30, ge=0.05, le=0.50),
    high_thresh: float = Query(0.60, ge=0.51, le=0.90)
):
    """
    Accepts a patient cohort CSV file, executes multi-disease predictions for each record,
    and returns a structured JSON summary along with the enriched CSV dataset.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid .csv file")
        
    try:
        contents = await file.read()
        results, enriched_csv, summary = batch_service.process_batch_csv(contents, low_thresh, high_thresh)
        return {
            'filename': file.filename,
            'summary': summary,
            'records': results,
            'enriched_csv': enriched_csv
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing batch CSV: {str(e)}")


@router.get("/sample-template")
def get_sample_template():
    """Returns downloadable CSV template with pre-populated schema and sample rows."""
    template_csv = batch_service.get_sample_csv_template()
    return Response(
        content=template_csv,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=healthpredict_batch_template.csv"}
    )

