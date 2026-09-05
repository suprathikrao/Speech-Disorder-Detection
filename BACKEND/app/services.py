"""
Core Services Layer: Bridges HTTP routes to ML pipeline and database persistence.
B.Tech Major Project - Dept. of Information Technology
"""

import os
import sys
import json
import glob
import uuid
import datetime
from sqlalchemy.orm import Session

# Add project root to sys.path
from app.config import BASE_DIR, DATA_DIR, MODELS_DIR, UPLOADS_DIR
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.predict import predict_audio
from src.train import train_models
from src.generate_mock_data import generate_dataset
from app.models import Patient, ScreeningRecord, ModelBenchmark


def perform_audio_screening(
    file_path: str,
    patient_name: str = None,
    patient_age: int = None,
    patient_gender: str = None,
    notes: str = None,
    db: Session = None
) -> dict:
    """
    Runs ML inference on the audio file, saves patient and screening record to MySQL,
    and returns comprehensive screening response.
    """
    # 1. Run core inference
    result = predict_audio(file_path=file_path, models_dir=str(MODELS_DIR))

    # 2. Persist to database if db session provided
    patient_obj = None
    screening_record = None

    if db:
        # Create or find patient
        if patient_name and patient_name.strip():
            patient_code = f"PT-{uuid.uuid4().hex[:8].upper()}"
            patient_obj = Patient(
                patient_code=patient_code,
                full_name=patient_name.strip(),
                age=patient_age,
                gender=patient_gender,
                notes=notes
            )
            db.add(patient_obj)
            db.flush()

        # Save screening record
        screening_record = ScreeningRecord(
            patient_id=patient_obj.id if patient_obj else None,
            audio_filename=os.path.basename(file_path),
            audio_path=file_path,
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            probabilities_json=json.dumps(result["probabilities"]),
            features_json=json.dumps(result["all_features"]),
            model_used=result["model_used"],
            screening_note=result["preliminary_screening_note"],
            created_at=datetime.datetime.utcnow()
        )
        db.add(screening_record)
        db.commit()
        db.refresh(screening_record)

        result["id"] = screening_record.id
        result["created_at"] = screening_record.created_at.isoformat()
        if patient_obj:
            result["patient"] = {
                "id": patient_obj.id,
                "patient_code": patient_obj.patient_code,
                "full_name": patient_obj.full_name,
                "age": patient_obj.age,
                "gender": patient_obj.gender
            }

    # Audio playback URL
    result["audio_url"] = f"/api/audio/{os.path.basename(file_path)}"
    return result


def execute_model_training(db: Session = None) -> dict:
    """
    Executes training pipeline across SVM, Random Forest, Logistic Regression.
    Updates MySQL ModelBenchmark records.
    """
    metadata = train_models(
        data_dir=str(DATA_DIR),
        features_csv=str(DATA_DIR / "features.csv"),
        models_dir=str(MODELS_DIR)
    )

    best_name = metadata.get("best_model")
    benchmarks = metadata.get("models_benchmark", {})

    if db:
        # Clear old benchmarks or append
        for m_name, metrics in benchmarks.items():
            bm = ModelBenchmark(
                model_name=m_name,
                accuracy=metrics["accuracy"],
                precision_score=metrics["precision"],
                recall_score=metrics["recall"],
                f1_score=metrics["f1_score"],
                confusion_matrix_json=json.dumps(metrics["confusion_matrix"]),
                is_best=(m_name == best_name),
                trained_at=datetime.datetime.utcnow()
            )
            db.add(bm)
        db.commit()

    return metadata


def populate_mock_data_if_needed(samples_per_class: int = 15) -> int:
    """Populates synthetic speech data into data/ folder if empty."""
    return generate_dataset(output_dir=str(DATA_DIR), samples_per_class=samples_per_class)


def count_data_samples() -> dict:
    """Counts available .wav samples per disorder class in data/."""
    counts = {}
    if os.path.exists(DATA_DIR):
        for entry in os.listdir(DATA_DIR):
            folder = os.path.join(DATA_DIR, entry)
            if os.path.isdir(folder):
                wavs = glob.glob(os.path.join(folder, "*.wav"))
                counts[entry] = len(wavs)
    return counts
