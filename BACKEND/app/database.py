"""
Database Session & Connection Management
B.Tech Major Project - Dept. of Information Technology
"""

import os
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import DATABASE_URL, MYSQL_SERVER_URL, DB_NAME, BACKEND_DIR

Base = declarative_base()


def init_mysql_database():
    """Ensure database exists on MySQL server before connecting to the specific database."""
    try:
        server_engine = create_engine(MYSQL_SERVER_URL, isolation_level="AUTOCOMMIT")
        with server_engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"))
        server_engine.dispose()
        print(f"MySQL database '{DB_NAME}' verified / initialized successfully.")
        return True
    except Exception as e:
        print(f"Notice: Direct MySQL database initialization encountered: {e}")
        return False


# Attempt MySQL initialization
mysql_ready = init_mysql_database()

try:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False
    )
    # Test connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print(f"Connected successfully to primary database at {DATABASE_URL}")
except Exception as err:
    print(f"Warning: Primary database connection failed ({err}). Falling back to local SQLite...")
    if os.getenv("VERCEL"):
        sqlite_path = Path("/tmp") / "speech_disorder_local.db"
    else:
        sqlite_path = BACKEND_DIR / "speech_disorder_local.db"
    engine = create_engine(f"sqlite:///{sqlite_path}", connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency to yield database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
