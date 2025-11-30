# =============================================================================
# AI Agent Performance Intelligence System - Makefile
# Course: DATA 230 (Data Visualization) at SJSU
# =============================================================================

.PHONY: help install train test lint format deploy build run stop logs clean monitor

# Default target
help:
	@echo "AI Agent Performance Intelligence System"
	@echo "========================================"
	@echo ""
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make train      - Train/retrain ML models"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linters"
	@echo "  make format     - Format code"
	@echo "  make run        - Run API locally"
	@echo "  make run-dev    - Run API with hot reload"
	@echo "  make build      - Build Docker image"
	@echo "  make deploy     - Build and deploy with Docker Compose"
	@echo "  make stop       - Stop all services"
	@echo "  make logs       - View logs"
	@echo "  make monitor    - Start monitoring dashboard"
	@echo "  make clean      - Clean up artifacts"
	@echo ""

# =============================================================================
# Development Setup
# =============================================================================

install:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

install-dev: install
	@echo "Installing development dependencies..."
	pip install black isort flake8 mypy pre-commit
	pre-commit install
	@echo "Development setup complete!"

# =============================================================================
# Model Training
# =============================================================================

train:
	@echo "Training ML models..."
	@echo "Step 1: Feature Engineering"
	cd notebooks && jupyter nbconvert --to notebook --execute 01_advanced_feature_engineering.ipynb --output 01_advanced_feature_engineering_executed.ipynb
	@echo "Step 2: Strategic Analysis"
	cd notebooks && jupyter nbconvert --to notebook --execute 02_strategic_analysis.ipynb --output 02_strategic_analysis_executed.ipynb
	@echo "Step 3: Model Training"
	cd notebooks && jupyter nbconvert --to notebook --execute 03_model_training_optimization.ipynb --output 03_model_training_optimization_executed.ipynb
	@echo "Model training complete! Models saved to models/"

train-quick:
	@echo "Running quick model training (notebooks only)..."
	python -c "import subprocess; subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', 'notebooks/03_model_training_optimization.ipynb'])"

# =============================================================================
# Testing
# =============================================================================

test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "Tests complete! Coverage report in htmlcov/"

test-fast:
	@echo "Running fast tests..."
	pytest tests/ -v -x --tb=short

test-api:
	@echo "Testing API endpoints..."
	pytest tests/test_api.py -v

# =============================================================================
# Code Quality
# =============================================================================

lint:
	@echo "Running linters..."
	flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503
	mypy src/ --ignore-missing-imports

format:
	@echo "Formatting code..."
	black src/ tests/ --line-length=120
	isort src/ tests/ --profile black

check: lint test
	@echo "All checks passed!"

# =============================================================================
# Local Development
# =============================================================================

run:
	@echo "Starting API server..."
	python run_api.py --host 0.0.0.0 --port 8000

run-dev:
	@echo "Starting API server with hot reload..."
	python run_api.py --host 127.0.0.1 --port 8000 --reload --debug

run-prod:
	@echo "Starting API server in production mode..."
	gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# =============================================================================
# Docker Operations
# =============================================================================

build:
	@echo "Building Docker image..."
	docker build -t agent-intelligence-api:latest .
	@echo "Docker image built successfully!"

build-no-cache:
	@echo "Building Docker image (no cache)..."
	docker build --no-cache -t agent-intelligence-api:latest .

deploy:
	@echo "Deploying with Docker Compose..."
	docker-compose up -d --build
	@echo "Deployment complete!"
	@echo "API available at http://localhost:8000"
	@echo "API docs at http://localhost:8000/docs"

deploy-dev:
	@echo "Deploying development environment..."
	docker-compose --profile dev up -d --build
	@echo "Development environment ready!"
	@echo "Dev API at http://localhost:8001"

stop:
	@echo "Stopping all services..."
	docker-compose down
	@echo "Services stopped."

restart:
	@echo "Restarting services..."
	docker-compose restart
	@echo "Services restarted."

logs:
	@echo "Showing logs..."
	docker-compose logs -f

logs-api:
	@echo "Showing API logs..."
	docker-compose logs -f api

# =============================================================================
# Monitoring
# =============================================================================

monitor:
	@echo "Starting monitoring..."
	@echo "Health check: http://localhost:8000/health"
	@echo ""
	@echo "Checking API health..."
	@curl -s http://localhost:8000/health | python -m json.tool || echo "API not running"
	@echo ""
	@echo "Container status:"
	docker-compose ps

health:
	@echo "Checking API health..."
	curl -s http://localhost:8000/health | python -m json.tool

metrics:
	@echo "Fetching metrics..."
	curl -s http://localhost:8000/metrics || echo "Metrics endpoint not available"

# =============================================================================
# Database Operations
# =============================================================================

db-init:
	@echo "Initializing database..."
	docker-compose exec postgres psql -U postgres -d agent_intelligence -f /docker-entrypoint-initdb.d/init.sql

db-migrate:
	@echo "Running database migrations..."
	alembic upgrade head

db-shell:
	@echo "Opening database shell..."
	docker-compose exec postgres psql -U postgres -d agent_intelligence

redis-shell:
	@echo "Opening Redis shell..."
	docker-compose exec redis redis-cli

# =============================================================================
# Cleanup
# =============================================================================

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "Cleanup complete!"

clean-docker:
	@echo "Cleaning Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	@echo "Docker cleanup complete!"

clean-all: clean clean-docker
	@echo "Full cleanup complete!"

# =============================================================================
# Documentation
# =============================================================================

docs:
	@echo "API documentation available at http://localhost:8000/docs"
	@echo "Opening in browser..."
	open http://localhost:8000/docs || xdg-open http://localhost:8000/docs || echo "Please open http://localhost:8000/docs manually"

# =============================================================================
# Utility
# =============================================================================

shell:
	@echo "Opening Python shell..."
	python -i -c "import pandas as pd; import numpy as np; print('pandas and numpy imported')"

notebook:
	@echo "Starting Jupyter notebook..."
	jupyter notebook notebooks/

version:
	@echo "Checking versions..."
	python --version
	pip show fastapi | grep Version
	pip show xgboost | grep Version
	docker --version || echo "Docker not installed"

