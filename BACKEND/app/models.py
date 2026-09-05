"""
SQLAlchemy ORM Models for Speech Disorder Detection System
B.Tech Major Project - Dept. of Information Technology
"""

import datetime
from sqlalchemy import Column, Integer, String, Float, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_code = Column(String(50), unique=True, index=True, nullable=False)
    full_name = Column(String(150), nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String(20), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    screenings = relationship("ScreeningRecord", back_populates="patient", cascade="all, delete-orphan")


class ScreeningRecord(Base):
    __tablename__ = "screening_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)
    audio_filename = Column(String(255), nullable=False)
    audio_path = Column(String(500), nullable=False)
    predicted_class = Column(String(100), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    probabilities_json = Column(Text, nullable=False)  # JSON string
    features_json = Column(Text, nullable=False)       # JSON string of extracted features
    model_used = Column(String(150), nullable=False)
    screening_note = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationship
    patient = relationship("Patient", back_populates="screenings")


class ModelBenchmark(Base):
    __tablename__ = "model_benchmarks"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_name = Column(String(150), nullable=False)
    accuracy = Column(Float, nullable=False)
    precision_score = Column(Float, nullable=False)
    recall_score = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    confusion_matrix_json = Column(Text, nullable=False)
    is_best = Column(Boolean, default=False)
    trained_at = Column(DateTime, default=datetime.datetime.utcnow)
