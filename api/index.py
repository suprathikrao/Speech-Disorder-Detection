"""
Vercel Serverless Function Entrypoint for FastAPI
Speech Disorder Detection API
B.Tech Major Project - Dept. of Information Technology
"""

from __future__ import annotations
import sys
from pathlib import Path

# Configure paths so Python resolves both app.* (from BACKEND) and src.* (from root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "BACKEND"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

try:
    from app.main import app
except Exception as err:
    # Graceful fallback application if backend dependencies fail during serverless cold start
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI(
        title="Speech Disorder Detection API (Recovery Mode)",
        version="1.0.0",
        description="Fallback mode due to backend initialization error."
    )

    @app.get("/")
    @app.get("/api")
    @app.get("/api/health")
    def initialization_error():
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Backend initialization failed",
                "detail": str(err)
            }
        )

# Ensure root '/' and '/api' routes are available for health monitors and index requests
existing_routes = [getattr(r, "path", None) for r in app.routes]

if "/" not in existing_routes:
    @app.get("/", tags=["Root"])
    def root_endpoint():
        return {
            "status": "online",
            "service": "Speech Disorder Detection API",
            "version": "1.0.0",
            "docs_url": "/docs",
            "health_url": "/api/health",
            "description": "Acoustic biomarker analysis and preliminary screening for speech disorders."
        }

if "/api" not in existing_routes:
    @app.get("/api", tags=["Root"])
    def api_endpoint():
        return {
            "status": "online",
            "service": "Speech Disorder Detection API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/api/health",
                "screenings": "/api/screenings",
                "upload_and_screen": "/api/screenings/upload-and-screen",
                "models_metrics": "/api/models/metrics",
                "models_train": "/api/models/train",
                "generate_mock": "/api/dataset/generate-mock",
                "docs": "/docs",
                "openapi": "/openapi.json"
            }
        }

# Standard ASGI handler export for Vercel Python runtime
handler = app
