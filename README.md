# Gold Risk ML Pipeline

End-to-end machine learning system that predicts gold market volatility
regimes and 5-day forward risk.

The project demonstrates production-style MLOps practices including
feature engineering, model training, an inference API, and containerized
deployment.

------------------------------------------------------------------------

## Project Overview

This repository demonstrates a production-style machine learning system
including:

-   automated data ingestion\
-   feature engineering pipeline\
-   machine learning training workflow\
-   model artifact management\
-   inference API\
-   containerized deployment

The system predicts:

-   **future market regime** (LOW / MEDIUM / HIGH volatility)
-   **5-day forward volatility risk**

------------------------------------------------------------------------

## Architecture

    Market Data (Stooq)
            │
            ▼
    Data Ingestion
            │
            ▼
    Feature Engineering
            │
            ▼
    Model Training
            │
            ▼
    Saved Model Artifacts
            │
            ▼
    FastAPI Inference Service
            │
            ▼
    Docker Container
            │
            ▼
    Prediction API + Demo UI

------------------------------------------------------------------------

## Quick Start

Run the project using Docker.

Build the container:

``` bash
docker build -t gold-risk-api .
```

Run the container:

``` bash
docker run -p 8080:8080 gold-risk-api
```

Access the API:

http://localhost:8080

Interactive API documentation:

http://localhost:8080/docs

------------------------------------------------------------------------

## Data Source

Market data is fetched programmatically from:

https://stooq.com

Raw market data is **not redistributed** in this repository.

Users are responsible for complying with the data provider's terms of
service.

------------------------------------------------------------------------

## Tech Stack

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   FastAPI
-   Docker

------------------------------------------------------------------------

## Project Purpose

This project was built as a portfolio MLOps system demonstrating how
machine learning models can be integrated into a reproducible and
deployable inference service.

The primary focus is on **engineering practices around ML systems**, not
only model training.
