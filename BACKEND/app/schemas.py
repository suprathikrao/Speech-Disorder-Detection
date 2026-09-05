"""
Pydantic Schemas for Request/Response Validation
B.Tech Major Project - Dept. of Information Technology
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


# Patient Schemas
class PatientCreate(BaseModel):
    patient_code: Optional[str] = None
    full_name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    notes: Optional[str] = None


class PatientResponse(BaseModel):
    id: int
    patient_code: str
    full_name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# Screening Schemas
class ScreeningResultResponse(BaseModel):
    id: Optional[int] = None
    audio_file: str
    audio_url: str
    predicted_class: str
    confidence: float
    confidence_percentage: str
    probabilities: Dict[str, float]
    model_used: str
    preliminary_screening_note: str
    key_indicators: Dict[str, float]
    all_features: Dict[str, float]
    disclaimer: str
    created_at: Optional[datetime] = None
    patient: Optional[PatientResponse] = None


class ScreeningRecordSummary(BaseModel):
    id: int
    patient_id: Optional[int] = None
    patient_name: Optional[str] = "Anonymous"
    audio_filename: str
    audio_url: str
    predicted_class: str
    confidence: float
    model_used: str
    created_at: datetime

    class Config:
        from_attributes = True


# Model Training & Metrics Schemas
class ModelMetricItem(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    is_best: bool


class TrainingResponse(BaseModel):
    status: str
    message: str
    timestamp: str
    best_model: str
    classes: List[str]
    total_samples: int
    train_samples: int
    test_samples: int
    benchmarks: List[ModelMetricItem]


# Health Check Schema
class HealthResponse(BaseModel):
    status: str
    database: str
    models_ready: bool
    data_samples: Dict[str, int]
    api_version: str
