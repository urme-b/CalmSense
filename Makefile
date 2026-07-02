.PHONY: help install install-dev test lint format experiment reproduce demo wesad data frontend frontend-build clean

help:
	@echo "CalmSense - subject-independent stress detection"
	@echo ""
	@echo "  install        Install the package"
	@echo "  install-dev    Install with dev tools"
	@echo "  experiment     Run the LOSO benchmark from raw WESAD"
	@echo "  reproduce      Regenerate every result and figure (full pipeline)"
	@echo "  demo           Run the calibration pipeline on synthetic data (no download)"
	@echo "  wesad          Download WESAD (~2 GB, primary dataset; needed for experiment/reproduce)"
	@echo "  data           Download PhysioNet Non-EEG (cross-dataset transfer only)"
	@echo "  frontend       Start the React dashboard"
	@echo "  test           Run tests"
	@echo "  lint           Lint with ruff"
	@echo "  format         Format with ruff"
	@echo "  clean          Remove caches and build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

experiment:
	python scripts/run_experiment.py

# Regenerate every number, figure, model, and the dashboard data in order
reproduce:
	python scripts/run_experiment.py
	python scripts/ablation.py
	python scripts/wrist.py
	python scripts/cross_dataset.py
	python scripts/calibration.py
	python scripts/personalize.py
	python scripts/fill_paper_tables.py
	python scripts/tuning.py
	python scripts/stats.py
	python scripts/export_onnx.py
	python scripts/build_dashboard_data.py

# Run the calibration pipeline on synthetic data, no dataset required
demo:
	python scripts/calibration.py --synthetic

# Download WESAD (~2 GB, research-only agreement) — the primary dataset for experiment/reproduce
wesad:
	python scripts/download_data.py --wesad

# Download the PhysioNet Non-EEG dataset for cross-dataset transfer
data:
	python scripts/download_data.py

test:
	pytest tests/ -q

lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

frontend:
	cd frontend && npm start

frontend-build:
	cd frontend && npm run build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/
