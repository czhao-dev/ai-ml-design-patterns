FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY app/ app/
COPY data/ data/

EXPOSE 8080

CMD ["python3", "-m", "app.main"]
