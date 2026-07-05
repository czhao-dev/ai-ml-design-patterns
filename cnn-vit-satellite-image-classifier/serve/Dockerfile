FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow and libGL (used by some TF ops)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps before copying source so this layer is cached
COPY serve/requirements.txt ./serve/requirements.txt
RUN pip install --no-cache-dir -r serve/requirements.txt

# Copy application source
COPY serve/ ./serve/
COPY src/   ./src/

# Model weights are mounted at runtime (see docker-compose.yml).
# Create an empty dir so the path exists even without a volume.
RUN mkdir -p models/trained

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "serve.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
