import sys
import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print(" [HealthPredict] Starting Full-Stack Platform")
    print("=" * 70)
    print(" * Web Application (SPA): http://localhost:8000")
    print(" * OpenAPI / Swagger UI:  http://localhost:8000/docs")
    print(" * Redoc Documentation:   http://localhost:8000/redoc")
    print("=" * 70)
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

