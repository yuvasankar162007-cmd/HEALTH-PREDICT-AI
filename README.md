# HealthPredict – Full-Stack Multi-Disease Risk & Lifestyle Analytics Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-0.28%2B-2A3B4C.svg)](https://www.uvicorn.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57.svg)](https://www.sqlite.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-green.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.52.0-blueviolet.svg)](https://shap.readthedocs.io/)

---

## 1. Project Overview & Problem Statement

Modern chronic diseases—such as **Type 2 Diabetes**, **Coronary Heart Disease**, and **Hypertension**—are among the leading contributors to global mortality and healthcare burdens. A significant proportion of these conditions are lifestyle-mediated and preventable if identified in early subclinical stages.

**HealthPredict** is an enterprise-grade, full-stack Machine Learning web application designed to screen and estimate individual risk probabilities across multiple lifestyle-related diseases simultaneously. By analyzing basic demographic variables, anthropometrics (with auto-calculated BMI), lifestyle habits (physical activity, sleep, smoking, alcohol, stress, diet), and standard clinical biomarkers (blood pressure, fasting glucose, cholesterol, heart rate), HealthPredict provides transparent, interpretable risk assessments, live counterfactual "What-If" simulations, and personalized wellness guidance.

> **Crucial Medical Scope:** HealthPredict is strictly an **educational and screening tool**. It does **NOT** diagnose diseases or prescribe medication. All predictions are probabilistic risk estimates intended to empower users to have informed discussions with qualified healthcare providers.

---

## 2. Full-Stack System Architecture

```
                                  ┌───────────────────────────────────────────────┐
                                  │           Modern SPA Frontend (Browser)       │
                                  │   Tailwind CSS • Plotly.js • Chart.js • Icons │
                                  └──────────────────────┬────────────────────────┘
                                                         │ HTTP / JSON REST Calls
                                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             FastAPI Backend (Port 8000)                                         │
│                                                                                                                 │
│   ├── /api/predict ────────► Multi-Disease Prediction Engine (XGBoost, RF, LR pipelines)                       │
│   ├── /api/explain ────────► SHAP Explainability Service (Waterfall attributions & plain-English narrative)     │
│   ├── /api/simulate ───────► What-If Counterfactual Intervention Simulator                                     │
│   ├── /api/batch ──────────► Multi-Record CSV Batch Ingestion & Enriched Export                                │
│   ├── /api/history ────────► Patient Assessment Persistence & Longitudinal Analytics (SQLite)                  │
│   ├── /api/models ─────────► Benchmark Metrics, Feature Metadata & ROC-AUC Comparisons                        │
│   └── /api/health ─────────► Operational Telemetry & Health Checks                                             │
└────────────────────────────────────────┬────────────────────────────────┬───────────────────────────────────────┘
                                         │                                │
                                         ▼                                ▼
                         ┌──────────────────────────────┐ ┌──────────────────────────────┐
                         │   SQLite Database            │ │ Serialized ML Models (Joblib)│
                         │   (data/healthpredict.db)    │ │ (models/*.pkl, metadata.json)│
                         └──────────────────────────────┘ └──────────────────────────────┘
```

---

## 3. Technology Stack

* **Backend REST API:** FastAPI, Uvicorn, Pydantic v2, SQLite
* **Frontend Application:** Modern Single-Page Application (SPA) with responsive Tailwind CSS, Lucide Icons, Plotly.js gauges/radars/waterfalls, and Chart.js longitudinal trends
* **Data Science & Machine Learning:** Pandas, NumPy, Scikit-Learn, XGBoost
* **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations)
* **Model Serialization:** Joblib
* **Automated Testing:** Pytest, FastAPI TestClient

---

## 4. Project Directory Structure

```
HealthPredict/
│
├── run_server.py                  # Single-command launcher for Full-Stack FastAPI + SPA
├── app.py                         # Streamlit prototype entrypoint (Backward Compatibility)
├── requirements.txt               # Pinned, tested dependencies
├── README.md                      # Comprehensive project documentation
├── .gitignore                     # Python / Data Science ignore rules
│
├── backend/                       # Production REST API Backend
│   ├── __init__.py
│   ├── main.py                    # FastAPI application, CORS, static file mounts & SPA router
│   ├── database.py                # SQLite database connection & CRUD operations
│   ├── schemas.py                 # Pydantic v2 validation and serialization models
│   ├── routes/                    # Modular API Route Controllers
│   │   ├── predict.py             # Single and multi-disease screening endpoints
│   │   ├── explain.py             # SHAP explainability endpoint
│   │   ├── simulate.py            # What-If counterfactual simulation endpoint
│   │   ├── history.py             # Assessment persistence and longitudinal stats
│   │   ├── models_info.py         # Benchmark tables and clinical metadata
│   │   └── batch.py               # Batch CSV upload and sample template download
│   └── services/                  # Business Logic & ML Service Layer
│       ├── ml_service.py          # Unified multi-model inference engine
│       ├── xai_service.py         # SHAP calculations and clinical narratives
│       ├── simulator_service.py   # Lifestyle modification projection engine
│       └── batch_service.py       # Cohort CSV processing and enriched dataset builder
│
├── frontend/                      # Modern Single-Page Application (SPA)
│   ├── index.html                 # Semantic SPA layout with 8 dedicated view panels
│   ├── css/
│   │   └── styles.css             # Healthcare SaaS design tokens, animations, print stylesheet
│   └── js/
│       ├── api.js                 # Asynchronous API client
│       ├── charts.js              # Plotly & Chart.js rendering engine
│       └── app.js                 # SPA router, live telemetry, form synchronization & state
│
├── data/
│   ├── healthpredict.db           # SQLite persistent database (created automatically)
│   ├── raw/                       # Authentic benchmark datasets (PIMA, Cleveland, Framingham)
│   └── processed/                 # Cleaned datasets with engineered features
│
├── models/                        # Serialized ML artifacts & benchmark metadata
│   ├── diabetes_pipeline.pkl      # Best trained pipeline for Diabetes (XGBoost)
│   ├── heart_pipeline.pkl         # Best trained pipeline for Heart Disease (Logistic Regression)
│   ├── hypertension_pipeline.pkl  # Best trained pipeline for Hypertension (Random Forest)
│   ├── feature_metadata.json      # Schema, types, units, and validation ranges
│   └── model_metrics.json         # Computed validation & test benchmark metrics
│
├── src/                           # Original ML pipeline training & engineering scripts
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── prediction.py
│   └── explainability.py
│
├── tests/
│   └── test_api.py                # Automated integration & unit test suite
│
└── utils/
    └── helpers.py                 # Clinical calculation utilities
```

---

## 5. REST API Documentation (OpenAPI / Swagger)

When the server is running, interactive API documentation is available at:
* **Swagger UI:** `http://localhost:8000/docs`
* **Redoc:** `http://localhost:8000/redoc`

### Key Endpoints:
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health and operational status |
| `POST` | `/api/predict/all` | Comprehensive screening across Diabetes, Heart Disease, and Hypertension |
| `POST` | `/api/predict/diabetes` | Isolated Type 2 Diabetes screening |
| `POST` | `/api/predict/heart` | Isolated Coronary Heart Disease screening |
| `POST` | `/api/predict/hypertension` | Isolated Hypertension screening |
| `GET` | `/api/predict/presets` | 1-Click authentic benchmark test personas |
| `POST` | `/api/explain/shap` | Local SHAP attribution breakdown and plain-English narrative |
| `POST` | `/api/simulate/what-if` | Counterfactual lifestyle intervention simulator |
| `POST` | `/api/batch/upload` | Upload multi-patient CSV cohort and retrieve enriched scores |
| `GET` | `/api/batch/sample-template` | Download sample CSV template |
| `GET` | `/api/history` | List historical assessment records with search and filters |
| `GET` | `/api/history/stats/summary` | Aggregated clinical telemetry and risk distribution |
| `GET` | `/api/history/{id}` | Retrieve individual assessment record by ID |
| `DELETE` | `/api/history/{id}` | Delete assessment record |
| `GET` | `/api/models/benchmarks` | Model evaluation comparison matrix (5 models per disease) |
| `GET` | `/api/models/metadata` | Input feature dictionary and validation bounds |

---

## 6. How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Full-Stack Platform
```bash
python run_server.py
```
* **Web Application (SPA):** Open your browser at [http://localhost:8000](http://localhost:8000)
* **Interactive API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Run Automated Tests
```bash
python -m pytest tests/test_api.py -v
```

---

## 7. Security, Privacy & Medical Limitations

* **100% Local Execution:** Zero remote cloud telemetry. All ML inference, SHAP calculations, and database records remain strictly on your local machine.
* **Database Privacy:** Patient assessments are stored in a local SQLite file (`data/healthpredict.db`) and can be deleted or cleared at any time.
* **Limitations:** Screening algorithms are trained on historical biomedical benchmark cohorts (PIMA, UCI Cleveland, Framingham) and do not replace laboratory blood panels, electrocardiograms, or physical clinical examinations.
