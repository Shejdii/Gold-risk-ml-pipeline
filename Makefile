PYTHON := python
PIP := $(PYTHON) -m pip

.PHONY: install format lint test clean ingest preprocess features train predict api pipeline all

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

format:
	$(PYTHON) -m black .

lint:
	-$(PYTHON) -m pylint --disable=R,C src

test:
	$(PYTHON) -m pytest -q

clean:
	$(PYTHON) -m scripts.clean

ingest:
	$(PYTHON) -m src.data.ingest

preprocess:
	$(PYTHON) -m src.data.preprocess

features:
	$(PYTHON) -m src.features.build_features

train:
	$(PYTHON) -m src.model.train_volatility_regime
	$(PYTHON) -m src.model.train_risk_score
	$(PYTHON) -m scripts.export_metrics

predict:
	$(PYTHON) -m src.model.predict

api:
	$(PYTHON) -m uvicorn src.api.api:app --reload

pipeline: ingest preprocess features train

all: pipeline