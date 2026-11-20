"""
HealthPredict Backend Application Entrypoint.
FastAPI REST API with Async Endpoints, OpenAPI / Swagger Docs, and SPA Static Hosting.
"""

import os
import sys
from contextlib import asynccontextmanager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.database import init_db
from backend.routes.predict import router as predict_router
from backend.routes.explain import router as explain_router
from backend.routes.simulate import router as simulate_router
from backend.routes.history import router as history_router
from backend.routes.models_info import router as models_router
from backend.routes.batch import router as batch_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context: initializes database tables and artifacts."""
    init_db()
    yield


app = FastAPI(
    title="HealthPredict API",
    description="Full-Stack Disease Risk Screening & Lifestyle Analytics REST API powered by Machine Learning & SHAP XAI.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Route Routers
app.include_router(predict_router)
app.include_router(explain_router)
app.include_router(simulate_router)
app.include_router(history_router)
app.include_router(models_router)
app.include_router(batch_router)


@app.get("/api/health", tags=["System"])
def health_check():
    """Returns the operational status of the HealthPredict API."""
    return {
        "status": "healthy",
        "service": "HealthPredict Multi-Disease AI Engine",
        "version": "2.0.0",
        "models_loaded": ["diabetes (XGBoost)", "heart (Logistic Regression)", "hypertension (Random Forest)"]
    }


# Frontend SPA Static Assets Mounting
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
if os.path.exists(FRONTEND_DIR):
    # Mount subfolders for static assets if they exist
    css_dir = os.path.join(FRONTEND_DIR, "css")
    js_dir = os.path.join(FRONTEND_DIR, "js")
    
    if os.path.exists(css_dir):
        app.mount("/css", StaticFiles(directory=css_dir), name="css")
    if os.path.exists(js_dir):
        app.mount("/js", StaticFiles(directory=js_dir), name="js")


@app.get("/", include_in_schema=False)
async def serve_spa_root():
    """Serves the Single Page Application index.html entrypoint."""
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse(
        status_code=404,
        content={"message": "Frontend index.html not found. Backend API is active at /docs."}
    )


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_spa_fallback(full_path: str):
    """Fallback route to serve static files or route client-side SPA navigation."""
    # If the path points to an actual file in frontend directory, serve it
    potential_file = os.path.join(FRONTEND_DIR, full_path)
    if os.path.isfile(potential_file):
        return FileResponse(potential_file)
        
    # Otherwise fallback to index.html for client-side routing
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse(
        status_code=404,
        content={"message": f"Path '{full_path}' not found. Visit /docs for API endpoints."}
    )

