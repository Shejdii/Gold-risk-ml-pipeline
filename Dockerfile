FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 🔥 KLUCZOWE
RUN python src/data/ingest.py && \
    python src/data/preprocess.py && \
    python src/features/build_features.py && \
    python src/model/train_volatility_regime.py && \
    python src/model/train_risk_score.py

RUN ls -R /app/artifacts && ls -R /app/data/features

CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8080"]