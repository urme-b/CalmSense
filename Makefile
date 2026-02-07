# =============================================================================
# CalmSense Makefile
# =============================================================================

.PHONY: help install install-dev test lint format train api frontend docker clean

# Default target
help:
	@echo "CalmSense - Multimodal Stress Detection System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run tests with coverage"
	@echo "  lint         Run linters (ruff, mypy)"
	@echo "  format       Format code with black and ruff"
	@echo "  train        Run full training pipeline"
	@echo "  api          Start FastAPI server"
	@echo "  frontend     Start React development server"
	@echo "  docker       Build and run Docker container"
	@echo "  docker-dev   Run development Docker container"
	@echo "  clean        Remove build artifacts"

# =============================================================================
# Installation
# =============================================================================

install:
	pip install --upgrade pip
	pip install -e .

install-dev:
	pip install --upgrade pip
	pip install -e ".[dev]"
	pip install pytest pytest-cov pytest-asyncio black ruff mypy
	cd frontend && npm install

# =============================================================================
# Testing
# =============================================================================

test:
	pytest tests/ -v --cov=src --cov=api --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -v -x --tb=short

test-unit:
	pytest tests/ -v -m "not integration" --tb=short

test-integration:
	pytest tests/ -v -m "integration" --tb=short

# =============================================================================
# Code Quality
# =============================================================================

lint:
	ruff check src/ api/ tests/
	mypy src/ api/ --ignore-missing-imports

format:
	black src/ api/ tests/
	ruff check --fix src/ api/ tests/

check: lint test

# =============================================================================
# Training & Experiments
# =============================================================================

train:
	python scripts/run_experiments.py --config config/experiment_config.yaml

train-ml:
	python scripts/run_experiments.py --models ml --cv loso

train-dl:
	python scripts/run_experiments.py --models dl --cv loso

evaluate:
	python scripts/evaluate_models.py --model-dir outputs/models

# =============================================================================
# Development Servers
# =============================================================================

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

api-prod:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

frontend:
	cd frontend && npm start

frontend-build:
	cd frontend && npm run build

# =============================================================================
# Docker
# =============================================================================

docker:
	docker-compose up --build

docker-build:
	docker build -t calmsense:latest .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/models/trained:/app/models/trained calmsense:latest

docker-dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build calmsense-dev

docker-stop:
	docker-compose down

docker-clean:
	docker-compose down -v --rmi local

# =============================================================================
# Documentation
# =============================================================================

docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8080

# =============================================================================
# Cleanup
# =============================================================================

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage
	rm -rf frontend/build frontend/node_modules/.cache

clean-all: clean
	rm -rf outputs/ models/trained/*.pkl models/checkpoints/
	rm -rf frontend/node_modules

# =============================================================================
# Utility
# =============================================================================

notebook:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

profile:
	python -m cProfile -o profile.stats scripts/run_experiments.py --profile

requirements:
	pip-compile requirements.in -o requirements.txt
	pip-compile requirements-dev.in -o requirements-dev.txt
