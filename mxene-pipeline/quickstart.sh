#!/bin/bash

# MXene Pipeline Quick Start Script

set -e

echo "=========================================="
echo "MXene Data Pipeline - Quick Start"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version"
echo ""

# Check if Ollama is installed
echo "Checking Ollama installation..."
if ! command -v ollama &> /dev/null; then
    echo "✗ Ollama not found"
    echo ""
    echo "Please install Ollama:"
    echo "  macOS: brew install ollama"
    echo "  Linux: curl -fsSL https://ollama.ai/install.sh | sh"
    echo "  Or visit: https://ollama.ai"
    exit 1
fi
echo "✓ Ollama installed"
echo ""

# Check if Ollama is running
echo "Checking Ollama service..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✗ Ollama not running"
    echo ""
    echo "Starting Ollama..."
    ollama serve > /dev/null 2>&1 &
    sleep 3
fi
echo "✓ Ollama is running"
echo ""

# Check if model is available
echo "Checking for llama3 model..."
if ! ollama list | grep -q "llama3"; then
    echo "✗ llama3 not found"
    echo ""
    echo "Downloading llama3 model (this may take a few minutes)..."
    ollama pull llama3
fi
echo "✓ llama3 model available"
echo ""

# Install dependencies
echo "Installing Python dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Setup environment
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "✓ .env created"
else
    echo "✓ .env already exists"
fi
echo ""

# Test the pipeline
echo "Testing pipeline components..."
echo ""

echo "1. Testing Ollama connection..."
python3 -c "
from langchain_community.llms import Ollama
llm = Ollama(model='llama3', base_url='http://localhost:11434')
response = llm.invoke('Say hello')
print('✓ Ollama responding correctly')
" || { echo "✗ Ollama test failed"; exit 1; }
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Run the full pipeline:"
echo "   python -m src.pipeline"
echo ""
echo "2. Or run individual steps:"
echo "   python -m src.pipeline --harvest-only    # Download papers"
echo "   python -m src.pipeline --extract-only    # Extract data"
echo "   python -m src.pipeline --aggregate-only  # Create CSV"
echo ""
echo "3. Or use Make commands:"
echo "   make run        # Full pipeline"
echo "   make harvest    # Download only"
echo "   make extract    # Extract only"
echo ""
echo "4. Add manual PDFs to: data/pdfs_manual/"
echo ""
echo "Output will be saved to: output_dataset.csv"
echo ""
