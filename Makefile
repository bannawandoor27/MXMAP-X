.PHONY: help install dev test lint format clean docker-up docker-down seed run

help:
	@echo "MXMAP-X Backend - Available Commands"
	@echo "===================================="
	@echo "install      - Install dependencies with Poetry"
	@echo "dev          - Install dev dependencies"
	@echo "test         - Run tests with pytest"
	@echo "lint         - Run linters (mypy, ruff)"
	@echo "format       - Format code with black"
	@echo "clean        - Remove cache and build files"
	@echo "docker-up    - Start Docker services"
	@echo "docker-down  - Stop Docker services"
	@echo "seed         - Generate and seed database"
	@echo "run          - Start development server"
	@echo "migrate      - Run database migrations"

install:
	poetry install --no-dev

dev:
	poetry install

test:
	poetry run pytest -v --cov=app --cov-report=term-missing

lint:
	poetry run mypy app
	poetry run ruff check app

format:
	poetry run black app tests scripts
	poetry run ruff check --fix app

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf htmlcov .coverage

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

seed:
	poetry run python scripts/generate_synthetic_data.py
	poetry run python scripts/seed_db.py

run:
	poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

migrate:
	poetry run alembic upgrade head

migrate-create:
	poetry run alembic revision --autogenerate -m "$(msg)"

train:
	poetry run python scripts/train_model.py

evaluate:
	poetry run python scripts/evaluate_model.py

ml-pipeline: seed train evaluate
	@echo "✓ ML pipeline completed!"
