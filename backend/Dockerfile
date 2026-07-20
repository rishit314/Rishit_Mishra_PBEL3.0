FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for OpenCV headless & ChromaDB
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 appuser

# Install Python dependencies directly (No PyTorch needed!)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend application files and pre-built vector database
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser knowledge_base ./knowledge_base
COPY --chown=appuser:appuser data ./data

USER appuser

EXPOSE 8000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "uvloop"]