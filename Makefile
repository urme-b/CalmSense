.PHONY: help install install-dev test lint format experiment api frontend frontend-build docker clean

help:
	@echo "CalmSense - subject-independent stress detection"
	@echo ""
	@echo "  install        Install the package"
	@echo "  install-dev    Install with dev tools"
	@echo "  experiment     Run the full LOSO benchmark from raw WESAD"
	@echo "  api            Start the prediction API"
	@echo "  frontend       Start the React dashboard"
	@echo "  test           Run tests"
	@echo "  lint           Lint with ruff"
	@echo "  format         Format with ruff"
	@echo "  docker         Build and run with Docker"
	@echo "  clean          Remove caches and build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

experiment:
	python scripts/run_experiment.py

test:
	pytest tests/ -q

lint:
	ruff check src/ api/ tests/ scripts/

format:
	ruff format src/ api/ tests/ scripts/
	ruff check --fix src/ api/ tests/ scripts/

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm start

frontend-build:
	cd frontend && npm run build

docker:
	docker compose up --build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/
