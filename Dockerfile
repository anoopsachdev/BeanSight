# ================================================================
# Coffee Bean Analyzer — Production Dockerfile
# ONNX-only inference image (~300MB, no PyTorch)
# ================================================================

FROM python:3.11-slim AS production

# System dependencies (libgomp needed by onnxruntime)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd -m -r -s /bin/bash appuser

WORKDIR /app

# Install Python dependencies (production only)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY model/predict.py ./model/predict.py
COPY model/__init__.py ./model/__init__.py

# Copy ONNX models (baked into image for Cloud Run)
COPY checkpoints/*.onnx ./checkpoints/

# Switch to non-root user
USER appuser

# Cloud Run uses PORT env var (default 8080)
ENV PORT=8080
ENV ENVIRONMENT=production
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# Start with uvicorn
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
