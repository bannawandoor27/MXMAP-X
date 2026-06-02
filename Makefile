.PHONY: help install dev test lint format clean \
        start stop restart logs status \
        db-reset db-migrate db-shell \
        data-synthetic data-custom train-synthetic train-custom train-retrain \
        evaluate models-clean models-backup \
        setup quickstart full-reset

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help:
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║         MXMAP-X - MXene Supercapacitor ML Platform        ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)🚀 Quick Start:$(NC)"
	@echo "  make quickstart          - Complete setup and start (recommended)"
	@echo "  make start               - Start all services"
	@echo "  make stop                - Stop all services"
	@echo ""
	@echo "$(GREEN)📊 Data & Training:$(NC)"
	@echo "  make data-synthetic      - Generate synthetic training data"
	@echo "  make data-custom         - Prepare for custom data (creates template)"
	@echo "  make train-synthetic     - Train models with synthetic data"
	@echo "  make train-custom        - Train models with custom data"
	@echo "  make train-retrain       - Retrain models (keeps existing data)"
	@echo "  make evaluate            - Evaluate trained models"
	@echo ""
	@echo "$(GREEN)🐳 Docker Services:$(NC)"
	@echo "  make start               - Start Docker services (db + api)"
	@echo "  make stop                - Stop Docker services"
	@echo "  make restart             - Restart Docker services"
	@echo "  make logs                - View service logs"
	@echo "  make status              - Check service status"
	@echo ""
	@echo "$(GREEN)🗄️  Database:$(NC)"
	@echo "  make db-reset            - Reset database (clear all data)"
	@echo "  make db-migrate          - Run database migrations"
	@echo "  make db-shell            - Open PostgreSQL shell"
	@echo ""
	@echo "$(GREEN)🤖 Models:$(NC)"
	@echo "  make models-clean        - Remove all trained models"
	@echo "  make models-backup       - Backup trained models"
	@echo ""
	@echo "$(GREEN)🧹 Maintenance:$(NC)"
	@echo "  make clean               - Remove cache and build files"
	@echo "  make full-reset          - Complete reset (db + models + data)"
	@echo "  make setup               - Initial project setup"
	@echo ""
	@echo "$(GREEN)🧪 Development:$(NC)"
	@echo "  make install             - Install dependencies"
	@echo "  make dev                 - Install dev dependencies"
	@echo "  make test                - Run tests"
	@echo "  make lint                - Run linters"
	@echo "  make format              - Format code"
	@echo ""
	@echo "$(YELLOW)📝 Examples:$(NC)"
	@echo "  make quickstart          # First time setup"
	@echo "  make train-synthetic     # Train with synthetic data"
	@echo "  make data-custom && make train-custom  # Use your own data"
	@echo ""

# ============================================================================
# Quick Start & Setup
# ============================================================================

quickstart: setup start data-synthetic db-migrate seed-db train-synthetic
	@echo "$(GREEN)✓ Quickstart complete!$(NC)"
	@echo "$(BLUE)→ API running at: http://localhost:8000$(NC)"
	@echo "$(BLUE)→ API docs at: http://localhost:8000/docs$(NC)"
	@echo "$(BLUE)→ Web UI at: http://localhost:8000/$(NC)"

setup:
	@echo "$(BLUE)Setting up project directories...$(NC)"
	@mkdir -p data models/cache logs
	@if [ ! -f .env ]; then cp .env.example .env; echo "$(GREEN)✓ Created .env file$(NC)"; fi
	@echo "$(GREEN)✓ Project setup complete$(NC)"

# ============================================================================
# Docker Services
# ============================================================================

start:
	@echo "$(BLUE)Starting Docker services...$(NC)"
	@docker-compose up 
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "$(YELLOW)Waiting for services to be ready...$(NC)"
	@sleep 5
	@make status

stop:
	@echo "$(BLUE)Stopping Docker services...$(NC)"
	@docker-compose down
	@echo "$(GREEN)✓ Services stopped$(NC)"

restart:
	@echo "$(BLUE)Restarting Docker services...$(NC)"
	@docker-compose restart
	@echo "$(GREEN)✓ Services restarted$(NC)"

logs:
	@docker-compose logs -f --tail=100

status:
	@echo "$(BLUE)Service Status:$(NC)"
	@docker-compose ps

# ============================================================================
# Data Generation
# ============================================================================

data-synthetic:
	@echo "$(BLUE)Generating synthetic training data...$(NC)"
	@docker-compose exec api python scripts/generate_synthetic_data.py
	@echo "$(GREEN)✓ Synthetic data generated: data/synthetic_training_data.csv$(NC)"
	@ls -lh data/synthetic_training_data.csv

data-custom:
	@echo "$(BLUE)Creating custom data template...$(NC)"
	@if [ ! -f data/custom_training_data.csv ]; then \
		echo "mxene_type,terminations,electrolyte,electrolyte_concentration,thickness_um,deposition_method,annealing_temp_c,annealing_time_min,interlayer_spacing_nm,specific_surface_area_m2g,pore_volume_cm3g,optical_transmittance,sheet_resistance_ohm_sq,areal_capacitance_mf_cm2,esr_ohm,rate_capability_percent,cycle_life_cycles,source,notes" > data/custom_training_data.csv; \
		echo "Ti3C2Tx,O,H2SO4,1.0,5.0,vacuum_filtration,120.0,60.0,1.2,98.5,0.12,75.0,45.0,350.5,2.5,85.0,10000,lab_experiment,Example data" >> data/custom_training_data.csv; \
		echo "$(GREEN)✓ Created template: data/custom_training_data.csv$(NC)"; \
		echo "$(YELLOW)→ Edit this file with your real data, then run: make train-custom$(NC)"; \
	else \
		echo "$(YELLOW)⚠ Template already exists: data/custom_training_data.csv$(NC)"; \
	fi

# ============================================================================
# Database Operations
# ============================================================================

db-reset:
	@echo "$(RED)⚠ This will delete ALL data in the database!$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to cancel, or Enter to continue...$(NC)"
	@read confirm
	@echo "$(BLUE)Resetting database...$(NC)"
	@docker-compose exec db psql -U mxmap_user -d mxmap_db -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
	@echo "$(GREEN)✓ Database reset complete$(NC)"

db-migrate:
	@echo "$(BLUE)Running database migrations...$(NC)"
	@docker-compose exec api alembic upgrade head
	@echo "$(GREEN)✓ Migrations complete$(NC)"

db-shell:
	@echo "$(BLUE)Opening PostgreSQL shell...$(NC)"
	@docker-compose exec db psql -U mxmap_user -d mxmap_db

seed-db:
	@echo "$(BLUE)Seeding database...$(NC)"
	@docker-compose exec api python scripts/seed_db.py
	@echo "$(GREEN)✓ Database seeded$(NC)"

# ============================================================================
# Model Training
# ============================================================================

train-synthetic: data-synthetic
	@echo "$(BLUE)Training models with synthetic data...$(NC)"
	@docker-compose exec api python scripts/train_model.py
	@echo "$(GREEN)✓ Training complete!$(NC)"
	@echo "$(BLUE)Models saved to: models/cache/$(NC)"
	@ls -lh models/cache/ | grep -E "\.json|\.joblib"

train-custom:
	@if [ ! -f data/custom_training_data.csv ]; then \
		echo "$(RED)✗ Custom data file not found!$(NC)"; \
		echo "$(YELLOW)→ Run 'make data-custom' first to create a template$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Training models with custom data...$(NC)"
	@cp data/custom_training_data.csv data/synthetic_training_data.csv
	@docker-compose exec api python scripts/train_model.py
	@echo "$(GREEN)✓ Training complete with custom data!$(NC)"
	@echo "$(BLUE)Models saved to: models/cache/$(NC)"
	@ls -lh models/cache/ | grep -E "\.json|\.joblib"

train-retrain:
	@echo "$(BLUE)Retraining models with existing data...$(NC)"
	@if [ ! -f data/synthetic_training_data.csv ]; then \
		echo "$(RED)✗ No training data found!$(NC)"; \
		echo "$(YELLOW)→ Run 'make data-synthetic' or 'make data-custom' first$(NC)"; \
		exit 1; \
	fi
	@docker-compose exec api python scripts/train_model.py
	@echo "$(GREEN)✓ Retraining complete!$(NC)"

evaluate:
	@echo "$(BLUE)Evaluating trained models...$(NC)"
	@docker-compose exec api python scripts/evaluate_model.py
	@echo "$(GREEN)✓ Evaluation complete$(NC)"

# ============================================================================
# Model Management
# ============================================================================

models-clean:
	@echo "$(RED)⚠ This will delete all trained models!$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to cancel, or Enter to continue...$(NC)"
	@read confirm
	@echo "$(BLUE)Removing trained models...$(NC)"
	@rm -rf models/cache/*
	@echo "$(GREEN)✓ Models removed$(NC)"

models-backup:
	@echo "$(BLUE)Backing up trained models...$(NC)"
	@mkdir -p models/backups
	@tar -czf models/backups/models_backup_$$(date +%Y%m%d_%H%M%S).tar.gz models/cache/
	@echo "$(GREEN)✓ Models backed up to: models/backups/$(NC)"
	@ls -lh models/backups/ | tail -1

# ============================================================================
# Development
# ============================================================================

install:
	@echo "$(BLUE)Installing dependencies...$(NC)"
	@poetry install --no-dev
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

dev:
	@echo "$(BLUE)Installing dev dependencies...$(NC)"
	@poetry install
	@echo "$(GREEN)✓ Dev dependencies installed$(NC)"

test:
	@echo "$(BLUE)Running tests...$(NC)"
	@poetry run pytest -v --cov=app --cov-report=term-missing

lint:
	@echo "$(BLUE)Running linters...$(NC)"
	@poetry run mypy app
	@poetry run ruff check app

format:
	@echo "$(BLUE)Formatting code...$(NC)"
	@poetry run black app tests scripts
	@poetry run ruff check --fix app
	@echo "$(GREEN)✓ Code formatted$(NC)"

clean:
	@echo "$(BLUE)Cleaning cache and build files...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf htmlcov .coverage
	@echo "$(GREEN)✓ Cleaned$(NC)"

# ============================================================================
# Complete Reset
# ============================================================================

full-reset: stop
	@echo "$(RED)⚠ This will delete ALL data, models, and reset the database!$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to cancel, or Enter to continue...$(NC)"
	@read confirm
	@echo "$(BLUE)Performing full reset...$(NC)"
	@rm -rf data/*.csv models/cache/* logs/*
	@docker-compose down -v
	@echo "$(GREEN)✓ Full reset complete$(NC)"
	@echo "$(YELLOW)→ Run 'make quickstart' to set up again$(NC)"

# ============================================================================
# Utility Commands
# ============================================================================

run-local:
	@echo "$(BLUE)Starting local development server...$(NC)"
	@poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

migrate-create:
	@if [ -z "$(msg)" ]; then \
		echo "$(RED)✗ Migration message required!$(NC)"; \
		echo "$(YELLOW)Usage: make migrate-create msg='your message'$(NC)"; \
		exit 1; \
	fi
	@poetry run alembic revision --autogenerate -m "$(msg)"

# ============================================================================
# Complete Workflows
# ============================================================================

workflow-synthetic: start db-reset db-migrate data-synthetic seed-db train-synthetic
	@echo "$(GREEN)✓ Complete synthetic workflow finished!$(NC)"
	@echo "$(BLUE)→ API: http://localhost:8000$(NC)"
	@echo "$(BLUE)→ Docs: http://localhost:8000/docs$(NC)"

workflow-custom: start db-reset db-migrate data-custom
	@echo "$(YELLOW)→ Edit data/custom_training_data.csv with your data$(NC)"
	@echo "$(YELLOW)→ Then run: make seed-db train-custom$(NC)"
