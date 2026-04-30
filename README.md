# Gold Risk ML Pipeline

End-to-end machine learning system for predicting gold market volatility regimes and 5-day forward risk.

This project focuses on building a reproducible ML system, not just training a model.

---

## 🌐 Live Demo

API is available at:

https://your-render-url.onrender.com

Example:
https://your-render-url.onrender.com/predict/latest


## ⚡ What this project does

The system:

- downloads and updates market data automatically
- builds time-based features from price series
- trains and compares multiple models
- selects the best model based on validation metrics
- stores artifacts and metrics for reproducibility
- exposes predictions via a FastAPI service
- runs inside a Docker container

---

## 🎯 Prediction targets

The system predicts:

- **volatility regime**: LOW / MEDIUM / HIGH  
- **5-day forward volatility risk** (regression)

---

## 🧠 Why this problem is non-trivial

Financial time series are:

- noisy and non-stationary
- weakly predictable
- sensitive to feature design

As a result:

- models are expected to perform only slightly better than baseline
- evaluation and validation strategy are critical

---

## 📊 Model performance (validation set)

### Classification (volatility regime)

| Model                | Accuracy | Macro F1 |
|---------------------|----------|----------|
| Dummy (baseline)    | 0.28     | 0.15     |
| Logistic Regression | 0.44     | 0.40     |
| Random Forest       | 0.41     | 0.40     |

Selected model: **Random Forest**



---

### Regression (5-day risk)

| Model              | RMSE     |
|-------------------|----------|
| Dummy (mean)      | 0.00531  |
| Ridge Regression  | 0.00483  |
| Random Forest     | 0.00486  |

Selected model: **Ridge**

---

## 🏗️ Pipeline architecture

```
Market Data (Yahoo Finance: GC=F)
        │
        ▼
Data Ingestion (daily update + caching)
        │
        ▼
Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Model Training + Evaluation
        │
        ▼
Model Selection
        │
        ▼
Artifacts + Metrics (saved to disk)
        │
        ▼
FastAPI Service
        │
        ▼
Dockerized API + Demo UI
```

---

## 🔁 Reproducible pipeline

Run full pipeline:

```bash
make all
```

Pipeline includes:

- data ingestion
- preprocessing
- feature generation
- model training
- metrics export

Artifacts:

```
artifacts/models/
artifacts/metrics/
data/features/
```

---

## 🚀 API

Run locally:

```bash
uvicorn src.api.api:app --reload
```

Run via Docker:

```bash
docker build -t gold-risk-api .
docker run -p 8080:8080 gold-risk-api
```

The same container is used for deployment on Render.

Endpoints:

- /predict/latest
- /health
- /docs

---

## ☁️ Deployment (Render)

The API is deployed as a live service using Render.

- containerized FastAPI application
- deployed directly from repository
- publicly accessible endpoint (demo)

This demonstrates:

- cloud deployment of ML inference services
- container-based deployment workflow
- separation between local development and production serving

---

## ⚠️ Limitations

- performance is close to baseline due to noisy financial data
- model should not be used for real trading decisions
- results depend heavily on feature design

## 🧱 Tech stack

- Python
- Pandas / NumPy
- Scikit-learn
- FastAPI
- Docker

---

## 🧠 What this project demonstrates

This is not just a model project. It demonstrates:

- reproducible ML pipelines
- baseline vs model comparison
- metric-driven model selection
- artifact management
- API deployment for inference

---

## ⚠️ Notes

- This is a portfolio/research system, not a trading system
- No claim of financial predictability or alpha
- Data is fetched from Yahoo Finance and not redistributed
