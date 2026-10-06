.PHONY: verify lint test audit build

verify: lint test build

lint:
	python -m ruff check aspire/

test:
	python -m pytest tests/ -v --tb=short --ignore=tests/integration --timeout=30

build:
	python -m build

audit:
	pip-audit --strict --desc || true
