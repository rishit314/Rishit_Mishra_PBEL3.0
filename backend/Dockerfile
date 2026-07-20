# =====================================================================
# Stage 1: Builder (Dependency Resolution & Wheel Compilation)
# =====================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment to isolate dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch CPU-only wheels first to leverage Docker layer caching
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining backend dependencies
# Ensure opencv-python-headless, fastapi, uvicorn, langchain, chromadb, redis are in requirements.txt
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =====================================================================
# Stage 2: Production Runner (Minimal Footprint)
# =====================================================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime libraries required for OpenCV headless & ChromaDB vector math
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 appuser

# Copy compiled virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy backend application source code and set ownership
COPY --chown=appuser:appuser backend/app ./app
COPY --chown=appuser:appuser backend/knowledge_base ./knowledge_base
COPY --chown=appuser:appuser backend/data ./data
# Switch to unprivileged user
USER appuser

# Expose application port
EXPOSE 8000

# Optimize Python runtime for container execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2

# Launch Uvicorn with asynchronous uvloop and worker scaling
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "uvloop"]