SHELL := cmd.exe
.SHELLFLAGS := /c
.PHONY: clean install format lint test features all

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

format:
	black .

lint:
	pylint --disable=R,C src || true

test:
	pytest -q

features:
	$(PYTHON) features\build_features.py

clean:
	if exist \features rmdir /s /q \features
	if exist artifacts rmdir /s /q artifacts

.PHONY: ingest preprocess features

ingest:
	$(PYTHON) src\data\ingest.py

preprocess:
	$(PYTHON) src\data\preprocess.py

features:
	$(PYTHON) features\build_features.py


all: install format lint test
