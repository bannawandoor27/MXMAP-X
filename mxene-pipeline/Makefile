.PHONY: help install setup test-ollama harvest extract aggregate run clean

help:
	@echo "MXene Data Pipeline - Available Commands:"
	@echo ""
	@echo "  make install      - Install Python dependencies"
	@echo "  make setup        - Initial setup (install + create .env)"
	@echo "  make test-ollama  - Test Ollama connection"
	@echo "  make harvest      - Download papers only"
	@echo "  make extract      - Extract data from PDFs only"
	@echo "  make aggregate    - Aggregate JSON results only"
	@echo "  make run          - Run full pipeline"
	@echo "  make clean        - Remove generated files"
	@echo ""

install:
	pip install -r requirements.txt

setup: install
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✓ Created .env file - please edit with your settings"; \
	else \
		echo "✓ .env already exists"; \
	fi
	@echo "✓ Setup complete"

test-ollama:
	@echo "Testing Ollama connection..."
	@python -c "from langchain_community.llms import Ollama; llm = Ollama(model='llama3'); print('✓ Ollama is running'); print('Model response:', llm.invoke('Hello')[:50])"

harvest:
	python -m src.pipeline --harvest-only

extract:
	python -m src.pipeline --extract-only

aggregate:
	python -m src.pipeline --aggregate-only

run:
	python -m src.pipeline

clean:
	rm -rf data/pdfs_downloaded/*.pdf
	rm -rf data/processed_json/*.json
	rm -rf data/logs/*.log
	rm -f output_dataset.csv
	@echo "✓ Cleaned generated files"
