# =============================================================================
# CalmSense Dockerfile - Multi-stage
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build Frontend
# -----------------------------------------------------------------------------
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci --silent

# Copy frontend source
COPY frontend/ ./

# Build production bundle
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 2: Python Backend
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_PATH=/app/models/trained \
    LOG_LEVEL=INFO

# Install system dependencies and
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY config/ ./config/

# Copy trained models (if
COPY models/ ./models/

# Copy frontend build from
COPY --from=frontend-build /app/frontend/build ./static

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
