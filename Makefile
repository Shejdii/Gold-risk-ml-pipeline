install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

format:
	black .

lint:
	pylint --disable=R,C src || true

test:
	pytest -q

all: install format lint test
