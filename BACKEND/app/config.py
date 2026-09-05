"""
Application Configuration
B.Tech Major Project - Dept. of Information Technology
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # d:\SDD
BACKEND_DIR = BASE_DIR / "BACKEND"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BACKEND_DIR / "uploads"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Database Configuration
# User provided: Root115:suprathik123@127.0.0.1:3306
DB_USER = os.getenv("DB_USER", "Root115")
DB_PASSWORD = os.getenv("DB_PASSWORD", "suprathik123")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "speech_disorder_db")

# Server-level URL (without database name, used to auto-create database if needed)
MYSQL_SERVER_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}"
# Database-level URL
DATABASE_URL = os.getenv("DATABASE_URL", f"{MYSQL_SERVER_URL}/{DB_NAME}")

# API Settings
API_TITLE = "Speech Disorder Detection API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Machine Learning based preliminary screening system for speech disorders (Dysarthria, Dysphonia, Stuttering, Normal)."
