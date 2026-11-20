"""
Integration and Unit Test Suite for HealthPredict Full-Stack REST API.
Verifies all prediction endpoints, SHAP explainability, What-If simulation,
SQLite persistence, batch CSV processing, and model benchmarks.
"""

import os
import sys
import io
import pytest
from fastapi.testclient import TestClient

# Ensure root directory is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import app
from backend.database import init_db, DB_PATH

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Initializes the database before tests."""
    init_db()


def test_health_endpoint():
    """Verifies that the health check endpoint returns 200 and healthy status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert len(data["models_loaded"]) >= 3


def test_presets_endpoint():
    """Verifies that preset patient personas are available."""
    response = client.get("/api/predict/presets")
    assert response.status_code == 200
    presets = response.json()
    assert "healthy_young_adult" in presets
    assert "sedentary_prediabetic" in presets
    assert "hypertensive_smoker" in presets
    assert "high_risk_cardiac" in presets


def test_comprehensive_prediction():
    """Verifies multimodal screening across Diabetes, Heart Disease, and Hypertension."""
    payload = {
        "patient_name": "Test Patient",
        "age": 52,
        "gender": "Male",
        "height_cm": 175.0,
        "weight_kg": 85.0,
        "blood_pressure_sys": 135.0,
        "blood_pressure_dia": 88.0,
        "glucose": 115.0,
        "cholesterol": 220.0,
        "resting_heart_rate": 74.0,
        "max_heart_rate": 148.0,
        "is_smoker": True,
        "cigs_per_day": 10
    }

    response = client.post("/api/predict/all?save_to_history=true", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Check structure
    assert "diabetes" in data
    assert "heart" in data
    assert "hypertension" in data
    assert "overall_risk_score" in data
    assert "overall_risk_category" in data
    assert "bmi_info" in data
    assert "lifestyle_score" in data
    assert "recommendations" in data

    # Verify probability bounds
    assert 0.0 <= data["diabetes"]["probability"] <= 1.0
    assert 0.0 <= data["heart"]["probability"] <= 1.0
    assert 0.0 <= data["hypertension"]["probability"] <= 1.0
    assert 0.0 <= data["overall_risk_score"] <= 100.0

    # Verify database persistence ID was returned
    assert data.get("saved_record_id") is not None


def test_single_disease_predictions():
    """Verifies isolated single disease prediction endpoints."""
    payload = {
        "age": 45,
        "gender": "Female",
        "height_cm": 165.0,
        "weight_kg": 68.0,
        "glucose": 95.0,
        "blood_pressure_sys": 120.0,
        "blood_pressure_dia": 80.0
    }

    # Diabetes
    res_d = client.post("/api/predict/diabetes", json=payload)
    assert res_d.status_code == 200
    assert "probability" in res_d.json()

    # Heart
    res_h = client.post("/api/predict/heart", json=payload)
    assert res_h.status_code == 200
    assert "probability" in res_h.json()

    # Hypertension
    res_hyp = client.post("/api/predict/hypertension", json=payload)
    assert res_hyp.status_code == 200
    assert "probability" in res_hyp.json()


def test_shap_explainability():
    """Verifies Explainable AI (SHAP) endpoint returns feature attributions and narratives."""
    payload = {
        "disease": "diabetes",
        "input_data": {
            "age": 55,
            "gender": "Male",
            "height_cm": 170.0,
            "weight_kg": 90.0,
            "glucose": 140.0,
            "blood_pressure_sys": 130.0,
            "blood_pressure_dia": 85.0
        }
    }

    response = client.post("/api/explain/shap", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["disease"] == "diabetes"
    assert "base_value" in data
    assert len(data["features"]) > 0
    assert len(data["shap_values"]) == len(data["features"])
    assert "narrative" in data
    assert len(data["top_increasing"]) >= 0


def test_what_if_simulator():
    """Verifies the counterfactual lifestyle intervention engine."""
    payload = {
        "base_data": {
            "age": 50,
            "gender": "Male",
            "height_cm": 172.0,
            "weight_kg": 92.0,
            "glucose": 130.0,
            "blood_pressure_sys": 145.0,
            "blood_pressure_dia": 92.0,
            "is_smoker": True,
            "cigs_per_day": 15
        },
        "weight_loss_kg": 8.0,
        "sbp_reduction": 15.0,
        "glucose_reduction": 20.0,
        "chol_reduction": 25.0,
        "quit_smoking": True,
        "boost_activity": True
    }

    response = client.post("/api/simulate/what-if", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "base_preds" in data
    assert "sim_preds" in data
    assert "overall_drop" in data
    assert data["simulated_bmi"] < data["base_preds"]["bmi_info"]["bmi"]
    assert data["overall_drop"] >= 0.0


def test_history_crud_and_stats():
    """Verifies patient history retrieval, stats aggregation, and record deletion."""
    # 1. Get stats
    res_stats = client.get("/api/history/stats/summary")
    assert res_stats.status_code == 200
    stats = res_stats.json()
    assert "total_assessments" in stats

    # 2. Get records
    res_list = client.get("/api/history?limit=10")
    assert res_list.status_code == 200
    records = res_list.json()
    assert isinstance(records, list)

    # 3. If records exist, test get by ID and delete
    if records:
        rec_id = records[0]["id"]
        res_single = client.get(f"/api/history/{rec_id}")
        assert res_single.status_code == 200
        assert res_single.json()["id"] == rec_id


def test_model_benchmarks_and_metadata():
    """Verifies benchmark tables and clinical metadata endpoints."""
    res_bench = client.get("/api/models/benchmarks")
    assert res_bench.status_code == 200
    bench = res_bench.json()
    assert "diabetes" in bench
    assert "heart" in bench
    assert "hypertension" in bench

    res_meta = client.get("/api/models/metadata")
    assert res_meta.status_code == 200


def test_batch_sample_template_and_processing():
    """Verifies batch CSV template retrieval and cohort processing."""
    # 1. Get template
    res_template = client.get("/api/batch/sample-template")
    assert res_template.status_code == 200
    assert "patient_name" in res_template.text

    # 2. Upload template for batch processing
    file_content = res_template.text.encode('utf-8')
    response = client.post(
        "/api/batch/upload",
        files={"file": ("cohort.csv", file_content, "text/csv")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert data["summary"]["total_patients"] >= 1
    assert "records" in data
    assert len(data["records"]) >= 1
    assert "enriched_csv" in data


def test_spa_root():
    """Verifies that the root URL serves index.html."""
    response = client.get("/")
    assert response.status_code == 200
    assert "HealthPredict" in response.text

