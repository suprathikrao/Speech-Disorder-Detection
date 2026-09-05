FROM python:3.11-slim

WORKDIR /app

# Install system audio libraries and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY BACKEND/requirements.txt ./BACKEND_requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r BACKEND_requirements.txt

# Copy source code and models
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start Uvicorn
CMD ["uvicorn", "BACKEND.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
