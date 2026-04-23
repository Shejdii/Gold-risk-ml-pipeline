FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -m src.data.ingest && \
    python -m src.data.preprocess && \
    python -m src.features.build_features && \
    python -m src.model.train_volatility_regime && \
    python -m src.model.train_risk_score && \
    python -m scripts.export_metrics

RUN ls -R /app/artifacts && ls -R /app/data/features

CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8080"]