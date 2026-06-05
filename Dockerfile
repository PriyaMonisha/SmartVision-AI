FROM python:3.11-slim

WORKDIR /app

# Rule 6: PYTHONPATH for src/ package imports
# PYTHONDONTWRITEBYTECODE: no .pyc files in image layers
# PYTHONUNBUFFERED: stdout/stderr immediately visible in docker logs
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Rule 29: system libs BEFORE pip install
# libgl1: Debian Bookworm name for OpenCV's libGL dep (was libgl1-mesa-glx on Bullseye)
# libglib2.0-0: GLib runtime required by OpenCV
# curl: required by compose healthcheck on the fastapi service
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in a separate layer for cache efficiency
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

# Copy application code after deps layer (code changes don't invalidate pip layer)
COPY . .

# Non-root user for security; create model cache dir with correct ownership
# BEFORE USER switch — the named volume model_cache:/app/models is created
# as root by Docker; mkdir here ensures appuser can write to it at runtime
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# NOTE: HEALTHCHECK intentionally omitted — this Dockerfile is reused for
# both the fastapi service (port 8000) and the streamlit service (port 8501).
# A port-8000 HEALTHCHECK would permanently mark the Streamlit container
# unhealthy. Per-service healthchecks are defined in docker-compose.yml.

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
