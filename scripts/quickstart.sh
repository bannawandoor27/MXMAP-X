#!/bin/bash

# MXMAP-X Backend Quick Start Script

set -e

echo "=========================================="
echo "MXMAP-X Backend Quick Start"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version"

# Check Poetry
if ! command -v poetry &> /dev/null; then
    echo "✗ Poetry not found. Please install Poetry first:"
    echo "  curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi
echo "✓ Poetry installed"

# Install dependencies
echo ""
echo "Installing dependencies..."
poetry install
echo "✓ Dependencies installed"

# Check PostgreSQL
echo ""
echo "Checking PostgreSQL..."
if command -v docker &> /dev/null; then
    echo "Starting PostgreSQL with Docker..."
    docker-compose up -d db
    echo "✓ PostgreSQL started"
    sleep 3
else
    echo "⚠ Docker not found. Please ensure PostgreSQL is running manually."
    echo "  Database: mxmap_db"
    echo "  User: mxmap_user"
    echo "  Password: mxmap_password"
fi

# Generate synthetic data
echo ""
echo "Generating synthetic training data..."
poetry run python scripts/generate_synthetic_data.py
echo "✓ Synthetic data generated"

# Seed database
echo ""
echo "Seeding database..."
poetry run python scripts/seed_db.py
echo "✓ Database seeded"

# Success message
echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Start the API server:"
echo "  poetry run uvicorn app.main:app --reload"
echo ""
echo "Or use make:"
echo "  make run"
echo ""
echo "API Documentation:"
echo "  http://localhost:8000/docs"
echo ""
echo "Health Check:"
echo "  curl http://localhost:8000/api/v1/health"
echo ""
