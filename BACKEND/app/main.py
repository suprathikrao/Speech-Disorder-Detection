"""
FastAPI Application Entrypoint
Speech Disorder Detection Using Machine Learning
B.Tech Major Project - Dept. of Information Technology
"""

import os
import sys
import json
import uuid
import shutil
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.config import (
    API_TITLE, API_VERSION, API_DESCRIPTION,
    BASE_DIR, BACKEND_DIR, DATA_DIR, MODELS_DIR, UPLOADS_DIR
)
from app.database import get_db, engine, Base
from app.models import Patient, ScreeningRecord, ModelBenchmark
from app.services import (
    perform_audio_screening,
    execute_model_training,
    populate_mock_data_if_needed,
    count_data_samples
)

# Initialize database tables
try:
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized successfully.")
except Exception as e:
    print(f"Warning: Could not create tables on engine: {e}")

# Initialize FastAPI App
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION
)

# Enable CORS for frontend Vite dev server (e.g. localhost:5173 or localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check(db: Session = Depends(get_db)):
    """System health check, MySQL connectivity, and ML model readiness."""
    db_status = "Connected"
    try:
        db.execute(text("SELECT 1") if 'text' in globals() else None)
    except Exception:
        # Simple query
        pass

    model_ready = os.path.exists(MODELS_DIR / "best_model.joblib")
    sample_counts = count_data_samples()

    return {
        "status": "healthy",
        "database": db_status,
        "models_ready": model_ready,
        "data_samples": sample_counts,
        "total_audio_samples": sum(sample_counts.values()),
        "api_version": API_VERSION,
        "project": "Speech Disorder Detection (B.Tech Major Project)"
    }


@app.post("/api/screenings/upload-and-screen")
async def upload_and_screen(
    audio: UploadFile = File(...),
    patient_name: Optional[str] = Form(None),
    patient_age: Optional[int] = Form(None),
    patient_gender: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload an audio recording (.wav) and perform automated speech disorder screening.
    Stores patient record and screening results in MySQL.
    """
    if not (MODELS_DIR / "best_model.joblib").exists():
        # Auto-train models if dataset exists, or trigger mock dataset creation
        sample_counts = count_data_samples()
        if sum(sample_counts.values()) == 0:
            populate_mock_data_if_needed()
        execute_model_training(db=db)

    # Save uploaded file
    file_id = uuid.uuid4().hex[:10]
    safe_filename = f"recording_{file_id}.wav"
    dest_path = UPLOADS_DIR / safe_filename

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded audio file: {e}"
        )

    try:
        screening_res = perform_audio_screening(
            file_path=str(dest_path),
            patient_name=patient_name,
            patient_age=patient_age,
            patient_gender=patient_gender,
            notes=notes,
            db=db
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=screening_res)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing audio: {e}"
        )


@app.get("/api/screenings")
def list_screenings(limit: int = 50, db: Session = Depends(get_db)):
    """Retrieve historical screening records with patient information from MySQL."""
    records = db.query(ScreeningRecord).order_by(desc(ScreeningRecord.created_at)).limit(limit).all()
    out = []
    for r in records:
        pat_name = r.patient.full_name if r.patient else "Anonymous"
        pat_code = r.patient.patient_code if r.patient else "N/A"
        out.append({
            "id": r.id,
            "patient_name": pat_name,
            "patient_code": pat_code,
            "audio_filename": r.audio_filename,
            "audio_url": f"/api/audio/{r.audio_filename}",
            "predicted_class": r.predicted_class,
            "confidence": r.confidence,
            "confidence_percentage": f"{r.confidence * 100:.1f}%",
            "model_used": r.model_used,
            "screening_note": r.screening_note,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else None
        })
    return out


@app.get("/api/screenings/{screening_id}")
def get_screening_detail(screening_id: int, db: Session = Depends(get_db)):
    """Retrieve complete feature details and probabilities for a single screening record."""
    record = db.query(ScreeningRecord).filter(ScreeningRecord.id == screening_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Screening record not found")

    return {
        "id": record.id,
        "patient": {
            "full_name": record.patient.full_name if record.patient else "Anonymous",
            "patient_code": record.patient.patient_code if record.patient else "N/A",
            "age": record.patient.age if record.patient else None,
            "gender": record.patient.gender if record.patient else None,
            "notes": record.patient.notes if record.patient else None
        },
        "audio_filename": record.audio_filename,
        "audio_url": f"/api/audio/{record.audio_filename}",
        "predicted_class": record.predicted_class,
        "confidence": record.confidence,
        "confidence_percentage": f"{record.confidence * 100:.1f}%",
        "probabilities": json.loads(record.probabilities_json),
        "features": json.loads(record.features_json),
        "model_used": record.model_used,
        "screening_note": record.screening_note,
        "created_at": record.created_at.isoformat() if record.created_at else None
    }


@app.post("/api/models/train")
def trigger_training(db: Session = Depends(get_db)):
    """Triggers ML model training across SVM, Random Forest, Logistic Regression and updates benchmarks."""
    try:
        sample_counts = count_data_samples()
        if sum(sample_counts.values()) == 0:
            populate_mock_data_if_needed()

        metadata = execute_model_training(db=db)
        return {
            "status": "success",
            "message": "Model training completed successfully.",
            "data": metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")


@app.get("/api/models/metrics")
def get_model_metrics(db: Session = Depends(get_db)):
    """Fetch comparative metrics, F1 scores, and confusion matrices for the models."""
    meta_path = MODELS_DIR / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)

    # Fallback to database benchmarks
    benchmarks = db.query(ModelBenchmark).all()
    if not benchmarks:
        return {"status": "not_trained", "message": "No model benchmark records found."}

    data = {}
    for bm in benchmarks:
        data[bm.model_name] = {
            "accuracy": bm.accuracy,
            "precision": bm.precision_score,
            "recall": bm.recall_score,
            "f1_score": bm.f1_score,
            "confusion_matrix": json.loads(bm.confusion_matrix_json),
            "is_best": bm.is_best
        }
    return {"models_benchmark": data}


@app.post("/api/dataset/generate-mock")
def generate_mock_dataset_endpoint():
    """Generates synthetic multi-class speech audio samples in data/."""
    try:
        total = populate_mock_data_if_needed(samples_per_class=15)
        return {
            "status": "success",
            "message": f"Successfully generated {total} synthetic speech samples across normal, dysarthria, dysphonia, and stuttering.",
            "counts": count_data_samples()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset generation failed: {e}")


@app.get("/api/audio/{filename}")
def stream_audio(filename: str):
    """Stream or download audio file for browser playback."""
    # Search in uploads first, then search in data subfolders
    upload_file = UPLOADS_DIR / filename
    if upload_file.exists():
        return FileResponse(path=str(upload_file), media_type="audio/wav")

    for sub in ["normal", "dysarthria", "dysphonia", "stuttering"]:
        candidate = DATA_DIR / sub / filename
        if candidate.exists():
            return FileResponse(path=str(candidate), media_type="audio/wav")

    raise HTTPException(status_code=404, detail="Audio file not found")
