.PHONY: install test lint synth

install:
	python3 -m pip install -e ".[dev]"
	cd infra && npm install

test:
	PYTHONPATH=src python3 -m pytest tests -q

lint:
	PYTHONPATH=src python3 -m ruff check src tests

synth:
	cd infra && npx cdk synth
