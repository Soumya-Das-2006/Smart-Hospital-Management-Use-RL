# ── Smart Hospital Resource Allocator — OpenEnv Dockerfile ───────────────────
# Base image: slim Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV (fast Python package installer)
RUN pip install --no-cache-dir uv

# Copy dependency files first (layer caching)
COPY pyproject.toml .
COPY requirements.txt .

# Install Python dependencies via UV
RUN uv pip install --system --no-cache -r requirements.txt

# Copy full project
COPY . .

# Set Python path so core.* and app.* imports resolve
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default environment variables (inference.py reads these)
ENV API_BASE_URL="https://api-inference.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
# HF_TOKEN must be set at runtime — no default

# Expose Flask port
EXPOSE 7860

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start with Gunicorn (production-grade WSGI server)
# HF Spaces uses port 7860 by default
CMD ["gunicorn", "run:app", "--workers", "1", "--bind", "0.0.0.0:7860", "--timeout", "180", "--log-level", "info"]
