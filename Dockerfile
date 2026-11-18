FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml ./
COPY app ./app
RUN pip install ".[dev]"

COPY docs ./docs
COPY tests ./tests
COPY data/raw ./data/raw

CMD ["sh", "-c", "python -m app.data.pipeline && exec uvicorn app.api:app_factory --factory --host 0.0.0.0 --port 8000"]
