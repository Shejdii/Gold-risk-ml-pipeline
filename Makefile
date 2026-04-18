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
	$(PYTHON) scripts/clean.py

ingest:
	$(PYTHON) src/data/ingest.py

preprocess:
	$(PYTHON) src/data/preprocess.py

features:
	$(PYTHON) src/features/build_features.py

train:
	$(PYTHON) src/model/train_volatility_regime.py
	$(PYTHON) src/model/train_risk_score.py

predict:
	$(PYTHON) src/model/predict.py

api:
	$(PYTHON) -m uvicorn src.api.api:app --reload

pipeline: ingest preprocess features train

all: pipeline