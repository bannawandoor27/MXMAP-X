# MXMAP-X — MXene Supercapacitor Prediction & Design Platform

> **ML-powered prediction, optimization, and process-aware design for MXene microsupercapacitors.**  
> FastAPI backend · XGBoost models · Interactive web UI · AC-line filtering · Printing process design · Literature extraction pipeline

![MXMAP-X End-to-End Workflow](workflow_video/workflow.gif)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Quick Start](#3-quick-start)
4. [Configuration](#4-configuration)
5. [API Reference](#5-api-reference)
6. [Machine Learning Pipeline](#6-machine-learning-pipeline)
7. [Web Interface](#7-web-interface)
8. [Feature Modules](#8-feature-modules)
9. [Database Schema](#9-database-schema)
10. [Testing](#10-testing)
11. [Docker Deployment](#11-docker-deployment)
12. [Literature Extraction Pipeline](#12-literature-extraction-pipeline)
13. [Troubleshooting](#13-troubleshooting)
14. [Development Reference](#14-development-reference)

---

## 1. Project Overview

MXMAP-X is a full-stack materials informatics platform for designing and optimizing MXene-based microsupercapacitors (MSCs). It combines:

- **ML prediction** of electrochemical performance (capacitance, ESR, rate capability, cycle life) with 95% confidence intervals
- **Multi-objective optimization** using Pareto-frontier search
- **Chemistry space exploration** via UMAP dimensionality reduction
- **AC-line filtering design** with EIS surrogate models for on-chip MSC layout
- **Printing process design** for gravure, screen, inkjet, slot-die, and doctor-blade deposition
- **Electrochromic visualization** of voltage-dependent color transitions
- **Literature extraction pipeline** using LLM-powered PDF parsing (Ollama + llama3)

### Model Performance

| Target | R² | RMSE | CI Coverage |
|--------|-----|------|-------------|
| Areal Capacitance | 0.952 | 24.3 mF/cm² | 94.2% |
| ESR | 0.888 | 0.34 Ω | 93.8% |
| Rate Capability | 0.823 | 4.9% | 95.5% |
| Cycle Life | 0.795 | 1824 cycles | 92.1% |

### Codebase Stats

- **~15,000+ lines** of Python
- **30+ Python files**, 7 HTML templates, 10+ test files
- **20+ API endpoints**, 4 ML models, 76 passing tests

---

## 2. Architecture

```
MXMAP-X/
├── app/
│   ├── main.py                    # FastAPI app, middleware, routers
│   ├── config.py                  # Settings (pydantic-settings)
│   ├── api/v1/endpoints/
│   │   ├── predictions.py         # /predict, /predict/batch
│   │   ├── devices.py             # /devices CRUD
│   │   ├── models.py              # /models/metrics, /health
│   │   ├── advanced.py            # /optimize, /explore, /compare, /recipes
│   │   ├── filtering.py           # /filtering/* (EIS surrogate)
│   │   └── printing.py            # /printing/* (process surrogate)
│   ├── core/
│   │   └── dependencies.py        # get_db, get_predictor
│   ├── db/
│   │   └── session.py             # SQLAlchemy engine (SQLite/PostgreSQL)
│   ├── ml/
│   │   ├── xgboost_model.py       # XGBoostPredictor (train/predict/persist)
│   │   ├── feature_engineer.py    # FeatureEngineer (encode/scale/impute)
│   │   ├── model_loader.py        # ModelLoader singleton
│   │   ├── optimization.py        # MultiObjectiveOptimizer (Pareto NSGA-II)
│   │   ├── eis_surrogate.py       # EISSurrogate (CPE circuit model)
│   │   └── printing_surrogates.py # PrintingSurrogate (percolation + Beer-Lambert)
│   ├── models/
│   │   ├── database.py            # SQLAlchemy ORM models
│   │   └── schemas.py             # Pydantic v2 request/response schemas
│   ├── templates/                 # Jinja2 HTML templates
│   └── web/routes.py              # Web UI routes (HTMLResponse)
├── mxene-pipeline/                # LLM literature extraction pipeline
│   ├── src/
│   │   ├── pipeline.py            # Orchestration
│   │   ├── downloader.py          # arXiv/DOI PDF download
│   │   ├── extractor.py           # LLM extraction (Ollama)
│   │   └── validator.py           # Schema + physics validation
│   ├── quickstart.sh
│   └── requirements.txt
├── scripts/
│   ├── generate_synthetic_data.py # 300 physics-informed training samples
│   ├── train_model.py             # Train XGBoost models
│   ├── evaluate_model.py          # Cross-validation + metrics
│   └── seed_db.py                 # Seed database from CSV
├── tests/                         # pytest suite (76 tests, all passing)
├── config/model_config.yaml       # ML hyperparameters
├── alembic/                       # Database migrations
└── docker-compose.yml
```

### Key Design Decisions

- **Singleton model loader**: `ModelLoader` caches the `XGBoostPredictor` in memory across requests
- **Quantile regression**: Separate XGBoost models for lower (2.5%), mean, and upper (97.5%) quantiles → 95% CI
- **SQLite/PostgreSQL duality**: `session.py` detects the URL scheme and applies `StaticPool` for SQLite or standard pooling for PostgreSQL
- **EIS surrogate**: Analytical `Rs + (CPE || R_leak)` circuit — no ML, pure physics, <1ms latency
- **Printing surrogate**: Percolation model + Beer-Lambert law + post-treatment corrections

---

## 3. Quick Start

### Prerequisites

```bash
python3 --version   # Python 3.10+
# Optional (for literature pipeline):
ollama --version    # macOS: brew install ollama
```

### Installation (Main API)

```bash
# Clone and enter
cd MXMAP-X

# Install dependencies
pip install -r requirements.txt          # or: poetry install

# Copy environment config
cp .env.example .env                     # edit DATABASE_URL if using PostgreSQL

# (PostgreSQL only) create database
createdb mxmap_db

# Generate synthetic training data and train models
python scripts/generate_synthetic_data.py
python scripts/train_model.py

# Seed database
python scripts/seed_db.py

# Start the server
uvicorn app.main:app --reload
```

### Installation (Literature Pipeline)

```bash
cd mxene-pipeline

# Option A — automated
./quickstart.sh

# Option B — manual
pip install -r requirements.txt
cp .env.example .env
ollama pull llama3

# Run extraction
python -m src.pipeline --max-papers 50
```

### Access Points

| URL | Description |
|-----|-------------|
| `http://localhost:8000/` | Main prediction interface |
| `http://localhost:8000/optimize` | Multi-objective optimization |
| `http://localhost:8000/explore` | Chemistry space UMAP |
| `http://localhost:8000/electrochromic` | Electrochromic visualization |
| `http://localhost:8000/filtering` | AC-line filtering design |
| `http://localhost:8000/printing` | Printing process design |
| `http://localhost:8000/recipes` | Recipe card export |
| `http://localhost:8000/docs` | Swagger UI |
| `http://localhost:8000/redoc` | ReDoc |

### Makefile Shortcuts

```bash
make run          # Start API with auto-reload
make test         # Run full pytest suite
make lint         # mypy + ruff
make format       # black + ruff --fix
make train        # Train XGBoost models
make evaluate     # Evaluate model performance
make ml-pipeline  # seed + train + evaluate
make seed         # Generate data and seed DB
make migrate      # Run alembic upgrade head
make docker-up    # docker-compose up -d
make docker-down  # docker-compose down
```

---

## 4. Configuration

### Environment Variables (`.env`)

```bash
# Database
DATABASE_URL=sqlite+aiosqlite:///./mxmap.db      # SQLite (dev/test)
# DATABASE_URL=postgresql+asyncpg://user:pw@localhost:5432/mxmap_db  # Postgres

DATABASE_ECHO=false

# API
API_V1_PREFIX=/api/v1
PROJECT_NAME=MXMAP-X Backend
VERSION=0.1.0
DEBUG=true

# CORS
BACKEND_CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]

# ML
MODEL_CONFIG_PATH=config/model_config.yaml
MODEL_CACHE_DIR=models/cache
```

### Model Configuration (`config/model_config.yaml`)

```yaml
xgboost:
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  quantiles: [0.025, 0.5, 0.975]

features:
  categorical: [mxene_type, terminations, electrolyte, deposition_method]
  numerical: [thickness_um, annealing_temp_c, annealing_time_min,
              interlayer_spacing_nm, specific_surface_area_m2g,
              electrolyte_concentration]

targets:
  - areal_capacitance_mf_cm2
  - esr_ohm
  - rate_capability_percent
  - cycle_life_cycles
```

---

## 5. API Reference

### Health & Info

```bash
GET  /api/v1/health              # {"status":"healthy","version":"0.1.0","database":"connected","model":"loaded"}
GET  /api/v1/models/metrics      # R², RMSE, training/test sizes
GET  /docs                       # Swagger UI
GET  /redoc                      # ReDoc
```

### Predictions

#### Single Prediction

```bash
POST /api/v1/predict
Content-Type: application/json

{
  "mxene_type": "Ti3C2Tx",
  "terminations": "O",
  "electrolyte": "H2SO4",
  "electrolyte_concentration": 1.0,
  "thickness_um": 5.0,
  "deposition_method": "vacuum_filtration",
  "annealing_temp_c": 120.0,
  "annealing_time_min": 60.0,
  "interlayer_spacing_nm": 1.2,
  "specific_surface_area_m2g": 98.5
}
```

**Response:**

```json
{
  "areal_capacitance": {"value": 350.5, "lower_ci": 320.0, "upper_ci": 381.0, "confidence": "high"},
  "esr":              {"value": 2.5,   "lower_ci": 2.1,   "upper_ci": 2.9,   "confidence": "medium"},
  "rate_capability":  {"value": 85.0,  "lower_ci": 80.0,  "upper_ci": 90.0,  "confidence": "medium"},
  "cycle_life":       {"value": 10000, "lower_ci": 8500,  "upper_ci": 11500, "confidence": "medium"},
  "overall_confidence": "high",
  "confidence_score": 0.92,
  "model_version": "v0.1.0",
  "prediction_time_ms": 15.3,
  "request_id": "req_abc123"
}
```

#### Batch Prediction

```bash
POST /api/v1/predict/batch
{"devices": [{...}, {...}]}   # max 100 devices
```

### Devices (Training Data)

```bash
GET  /api/v1/devices                          # List (paginated, filterable)
GET  /api/v1/devices/{id}                     # Single device
POST /api/v1/devices                          # Add training sample
```

**Add training data:**

```json
{
  "mxene_type": "Ti3C2Tx",
  "terminations": "O",
  "electrolyte": "H2SO4",
  "thickness_um": 5.0,
  "deposition_method": "vacuum_filtration",
  "areal_capacitance_mf_cm2": 350.5,
  "esr_ohm": 2.5,
  "rate_capability_percent": 85.0,
  "cycle_life_cycles": 10000,
  "source": "DOI:10.1234/example"
}
```

### Advanced Features

```bash
POST /api/v1/optimize            # Multi-objective Pareto optimization
GET  /api/v1/explore             # Chemistry space UMAP map
POST /api/v1/compare             # Compare 2–10 candidates side-by-side
GET  /api/v1/recipes/{id}        # Export fabrication recipe (JSON)
WS   /api/v1/ws/predict          # WebSocket real-time predictions
```

### AC-Line Filtering

```bash
POST /api/v1/filtering/predict   # Predict EIS KPIs for given geometry
POST /api/v1/filtering/optimize  # Optimize interdigitated layout
POST /api/v1/filtering/fit       # Fit CPE params from measured Bode CSV
GET  /api/v1/filtering/presets   # Preset geometries and process templates
```

**Filtering predict request:**

```json
{
  "frequency_range_hz": [1, 100000],
  "load_resistance_ohm": 33.0,
  "geometry": {
    "finger_width_um": 8.0,
    "finger_spacing_um": 8.0,
    "finger_length_um": 2000.0,
    "num_fingers_per_electrode": 50,
    "overlap_length_um": 1800.0,
    "thickness_nm": 200.0,
    "substrate": "Si/SiO2"
  },
  "mxene_film": {
    "porosity_pct": 30.0,
    "sheet_res_ohm_sq": 10.0,
    "electrolyte": "PVA/H2SO4",
    "process": "photolithography"
  }
}
```

**KPIs in response:**

| Field | Description |
|-------|-------------|
| `phase_deg_120hz` | Phase angle @ 120 Hz (target ≤ -80°) |
| `capacitance_mf_cm2_120hz` | Areal capacitance @ 120 Hz |
| `impedance_ohm_120hz` | Impedance magnitude @ 120 Hz |
| `attenuation_db_60hz` | Ripple attenuation (more negative = better) |
| `device_area_mm2` | Device footprint area |

### Printing Process Design

```bash
POST /api/v1/printing/recommend  # Recommend ink + predict film properties
POST /api/v1/printing/estimate   # Estimate from explicit ink formulation
POST /api/v1/printing/fit        # Fit surrogate from experimental CSV
GET  /api/v1/printing/presets    # Process parameters and typical targets
```

---

## 6. Machine Learning Pipeline

### Feature Engineering (`app/ml/feature_engineer.py`)

| Category | Features |
|----------|----------|
| Categorical (label-encoded) | `mxene_type`, `terminations`, `electrolyte`, `deposition_method` |
| Numerical (standardized) | `thickness_um`, `electrolyte_concentration`, `annealing_temp_c`, `annealing_time_min`, `interlayer_spacing_nm`, `specific_surface_area_m2g` |
| Derived | `thickness_um²`, `surface_area × pore_volume`, `has_annealing` (bool), `packing_density_proxy` |

### Model Architecture

```
XGBoostPredictor
├── models[target]["lower"]  → XGBRegressor(objective="reg:quantileerror", quantile=0.025)
├── models[target]["mean"]   → XGBRegressor(objective="reg:squarederror")
└── models[target]["upper"]  → XGBRegressor(objective="reg:quantileerror", quantile=0.975)
```

One triplet per target: `areal_capacitance`, `esr`, `rate_capability`, `cycle_life`.

### Training Scripts

```bash
# Generate 300 physics-informed samples
python scripts/generate_synthetic_data.py

# Train models (saves to models/cache/)
python scripts/train_model.py

# Evaluate with cross-validation
python scripts/evaluate_model.py
```

### Physics in Synthetic Data

The generator encodes known structure–property relationships:

| Effect | Impact |
|--------|--------|
| Thicker film | ↑ capacitance, ↓ rate capability |
| H₂SO₄ electrolyte | ↑ capacitance, ↓ cycle life |
| Ionic liquid | ↑ high-temp performance, ↑ ESR |
| Larger interlayer spacing | ↑ rate capability |
| Vacuum filtration | ↑ film quality |
| Measurement noise | ±10–15% on all metrics |
| Missing optical data | 10–15% dropout |

### Python Usage

```python
import asyncio
from app.ml.model_loader import get_predictor
from app.models.schemas import PredictionRequest

async def predict():
    predictor = get_predictor()
    req = PredictionRequest(
        mxene_type="Ti3C2Tx",
        terminations="O",
        electrolyte="H2SO4",
        thickness_um=5.0,
        deposition_method="vacuum_filtration",
    )
    result = await predictor.predict(req)
    print(f"Capacitance: {result.areal_capacitance.value} mF/cm²")
    print(f"Confidence:  {result.overall_confidence}")

asyncio.run(predict())
```

---

## 7. Web Interface

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Reactivity | Alpine.js 3.x |
| Styling | TailwindCSS 3.x |
| Charts | Plotly.js + Chart.js 4.x |
| HTTP | Axios 1.x |
| Templates | Jinja2 (FastAPI) |
| Server | FastAPI + Starlette |

### Color Scheme

```css
Primary:   #667eea  /* Purple */
Secondary: #764ba2  /* Dark Purple */
Success:   #10b981  /* Green  */
Warning:   #f59e0b  /* Yellow */
Error:     #ef4444  /* Red    */
```

### Confidence Badges

- 🟢 **High** (>90%): green
- 🟡 **Medium** (70–90%): yellow
- 🔴 **Low** (<70%): red

### Adding a New Page

1. Create `app/templates/mypage.html`:
   ```html
   {% extends "base.html" %}
   {% block content %}
     <!-- content -->
   {% endblock %}
   ```
2. Register route in `app/web/routes.py`:
   ```python
   @router.get("/mypage", response_class=HTMLResponse)
   async def mypage(request: Request):
       return templates.TemplateResponse(request, "mypage.html")
   ```
3. Add link in `base.html` nav.

---

## 8. Feature Modules

### 8.1 Supercapacitor Performance Prediction

The primary ML feature. Input: material composition + processing. Output: 4 targets with 95% CI and confidence level.

**Supported MXene types:** Ti₃C₂Tₓ, Mo₂CTₓ, V₂CTₓ, Nb₂CTₓ, Ti₂CTₓ  
**Electrolytes:** H₂SO₄, KOH, Na₂SO₄, EMIM-TFSI, PVA/H₃PO₄  
**Deposition methods:** vacuum filtration, spray coating, spin coating, drop casting, screen printing, inkjet

### 8.2 Multi-Objective Optimization

Pareto-frontier search using NSGA-II-inspired population sampling.

```json
POST /api/v1/optimize
{
  "objectives": [
    {"metric": "capacitance", "target": "maximize", "weight": 1.0},
    {"metric": "esr",         "target": "minimize", "weight": 0.8}
  ],
  "constraints": {
    "thickness_min": 2.0,
    "thickness_max": 15.0
  },
  "population_size": 100,
  "generations": 50
}
```

### 8.3 Chemistry Space Exploration

UMAP reduces the 17-feature space to 2D for visual exploration. Hover any point to see full material composition and predicted performance. Color by capacitance, ESR, rate capability, or cycle life.

### 8.4 Electrochromic Visualization

3D animation of voltage-dependent color transitions in a layered MXene device stack:

- **Oxidized state** (positive voltage): yellow/gold
- **Reduced state** (negative voltage): blue/green
- Includes cyclic voltammetry (CV) curve simulation with capacitive + faradaic currents

### 8.5 AC-Line Filtering Design

Physics-based EIS surrogate for designing on-chip MXene MSCs as ripple filters.

**Circuit model:** `Rs + (CPE || R_leak)`  
**CPE equation:** `Z_CPE = 1 / (Q · (jω)^α)`

**Geometry → circuit parameter mapping** (`geometry_to_params`):
- `Rs` = sheet resistance × (path length / finger width) + contact resistance
- `Q` = base capacitance (15 mF/cm²) × active area × porosity factor
- `α` = 0.95 − (porosity% / 100) × 0.15, clipped to [0.75, 0.95]

**Ripple attenuation** (shunt topology):
- `H(jω) = Z_MSC / (R_L + Z_MSC)`
- Higher `R_L` → more attenuation (more negative dB)

**Optimizer:** NSGA-II-style random search (population=60, 40 generations), sorted by minimum device footprint area.

### 8.6 Printing Process Design

Surrogate models for five printing processes:

| Process | Solids (wt%) | Viscosity (mPa·s) | Thickness/pass (nm) |
|---------|-------------|-------------------|---------------------|
| Gravure | 8–15 | 50–200 | 150 |
| Screen | 5–12 | 800–3000 | 300 |
| Inkjet | 0.5–3 | 5–20 | 50 |
| Slot-die | 3–10 | 100–500 | 200 |
| Doctor-blade | 5–15 | 200–1000 | 250 |

**Physical models:**
- Sheet resistance: percolation model (threshold at 30% coverage)
- Transmittance: Beer-Lambert law with haze correction
- Post-treatment: annealing (up to −40% Rs), pressing (up to −30% Rs)
- Manufacturability score: printability window (30 pts), clog/flood risk (25 pts), pass count (20 pts), post-treatment burden (15 pts), yield (10 pts)

### 8.7 Recipe Card System

Generate and export fully reproducible fabrication recipes as JSON. Includes material composition, processing parameters, predicted performance with CI, and step-by-step fabrication instructions.

### 8.8 WebSocket Predictions

```javascript
const ws = new WebSocket("ws://localhost:8000/api/v1/ws/predict");
ws.send(JSON.stringify({ mxene_type: "Ti3C2Tx", ... }));
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

---

## 9. Database Schema

### Tables

| Table | Purpose |
|-------|---------|
| `devices` | Training data (material + performance) |
| `predictions` | Prediction history with CI |
| `training_metadata` | Model version tracking |
| `filtering_models` | Fitted EIS CPE parameters |
| `filtering_runs` | Filtering prediction history |
| `printing_calibrations` | Per-process printing coefficients |
| `printing_runs` | Printing recommendation history |

### Key Fields: `devices`

```
mxene_type            VARCHAR   Ti3C2Tx, Mo2CTx, ...
terminations          VARCHAR   O, F, OH, mixed
electrolyte           VARCHAR   H2SO4, KOH, ...
thickness_um          FLOAT
deposition_method     VARCHAR
annealing_temp_c      FLOAT     nullable
annealing_time_min    FLOAT     nullable
interlayer_spacing_nm FLOAT     nullable
specific_surface_area_m2g FLOAT nullable
areal_capacitance_mf_cm2  FLOAT nullable
esr_ohm               FLOAT     nullable
rate_capability_percent   FLOAT nullable
cycle_life_cycles     INTEGER   nullable
source                VARCHAR   DOI or paper reference
```

### Migrations

```bash
alembic revision --autogenerate -m "description"
alembic upgrade head
alembic downgrade -1
```

---

## 10. Testing

```bash
# Run full suite (76 tests)
python3 -m pytest tests/ -v

# With coverage report
pytest --cov=app --cov-report=html

# Specific modules
pytest tests/test_health.py tests/test_predictions.py -v
pytest tests/test_xgboost_model.py -v
pytest tests/test_filtering.py tests/test_filtering_api.py -v
pytest tests/test_printing_api.py tests/test_advanced_endpoints.py -v

# Type checking
mypy app

# Linting
ruff check app
black --check app
```

### Test Modules

| File | Coverage |
|------|----------|
| `test_health.py` | Health endpoint, root HTML |
| `test_predictions.py` | Single/batch predict, validation |
| `test_xgboost_model.py` | FeatureEngineer, XGBoostPredictor train/predict/persist |
| `test_filtering.py` | EIS surrogate physics (CPE impedance, attenuation, geometry mapping) |
| `test_filtering_api.py` | Filtering API endpoints |
| `test_printing_api.py` | Printing surrogate, API endpoints |
| `test_advanced_endpoints.py` | Optimization, comparison |

### Test Configuration (`tests/conftest.py`)

- Uses **in-memory SQLite** (`sqlite+aiosqlite:///:memory:`) with `StaticPool`
- `ASGITransport` for async HTTP client (httpx)
- Database tables created fresh per session

---

## 11. Docker Deployment

```yaml
# docker-compose.yml summary
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      DATABASE_URL: postgresql+asyncpg://mxmap:mxmap@db:5432/mxmap_db
    depends_on: [db]

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: mxmap_db
      POSTGRES_USER: mxmap
      POSTGRES_PASSWORD: mxmap
    volumes: [postgres_data:/var/lib/postgresql/data]
```

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop
docker-compose down

# Full reset (wipe DB)
docker-compose down -v && docker-compose up -d
```

### Production Server

```bash
# Gunicorn with Uvicorn workers
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## 12. Literature Extraction Pipeline (`mxene-pipeline/`)

An LLM-powered pipeline that downloads MXene papers from arXiv/DOI, extracts experimental data using Ollama (llama3), validates it against physics constraints, and outputs a structured CSV for training.

### Pipeline Architecture

```
mxene-pipeline/
├── src/
│   ├── pipeline.py      # Orchestrator: download → extract → validate → aggregate
│   ├── downloader.py    # arXiv API + DOI resolver, PDF save
│   ├── extractor.py     # MXeneExtractor: PyMuPDF text + Ollama LLM
│   └── validator.py     # Schema check + physics range validation
├── data/
│   ├── pdfs_downloaded/ # Raw PDFs
│   ├── processed_json/  # Per-paper JSON
│   └── logs/
└── output_dataset.csv   # Final aggregated dataset
```

### Setup & Usage

```bash
cd mxene-pipeline
./quickstart.sh                          # Automated setup

# Or manually:
pip install -r requirements.txt
cp .env.example .env
ollama pull llama3                        # ~4 GB download

# Run pipeline
python -m src.pipeline --max-papers 50   # Extract from 50 papers
python -m src.pipeline --query "Ti3C2 supercapacitor" --max-papers 100
```

### Extracted Fields

```json
{
  "mxene_type": "Ti3C2Tx",
  "terminations": "O, OH",
  "electrolyte": "H2SO4",
  "electrolyte_concentration_M": 1.0,
  "thickness_um": 5.0,
  "deposition_method": "vacuum_filtration",
  "areal_capacitance_mf_cm2": 350.5,
  "esr_ohm": 2.5,
  "rate_capability_percent": 85.0,
  "cycle_life_cycles": 10000,
  "paper_doi": "10.1234/example",
  "paper_title": "...",
  "extraction_timestamp": "2024-01-15T10:00:00"
}
```

### Physics Validation Rules

```python
0   < areal_capacitance_mf_cm2 < 5000
0   < esr_ohm                  < 1000
0   < rate_capability_percent  < 100
0   < cycle_life_cycles        < 1000000
0.1 < thickness_um             < 1000
```

### Integration with Main App

```bash
# Copy extracted dataset to training location
cp mxene-pipeline/output_dataset.csv data/literature_data.csv

# Merge with synthetic data and retrain
python scripts/train_model.py --data data/literature_data.csv
```

### Programmatic API

```python
from src.extractor import MXeneExtractor
from src.validator import DataValidator
from src.downloader import PaperDownloader

# Download papers
downloader = PaperDownloader()
papers = downloader.search_arxiv("MXene microsupercapacitor Ti3C2", max_results=20)
for paper in papers:
    downloader.download_pdf(paper["pdf_url"], f"data/pdfs/{paper['arxiv_id']}.pdf")

# Extract data
extractor = MXeneExtractor()
result = extractor.extract_from_pdf("data/pdfs/paper.pdf")

# Validate
validator = DataValidator()
valid = [exp for exp in result.experiments if validator.validate(exp)]

# Search literature database (example FastAPI endpoint)
# GET /api/v1/literature/search?mxene_type=Ti3C2Tx&min_capacitance=200
```

### Scheduled Updates

```bash
# Weekly cron (Sundays at 2 AM)
(crontab -l; echo "0 2 * * 0 cd $(pwd)/mxene-pipeline && python -m src.pipeline --max-papers 20 >> data/logs/cron.log 2>&1") | crontab -
```

---

## 13. Troubleshooting

### API won't start

```bash
# Check Python version (need 3.10+)
python3 --version

# Check dependencies
pip install -r requirements.txt

# Check database connection (SQLite default works out of the box)
python3 -c "from app.db.session import engine; print('DB OK')"

# Check logs
uvicorn app.main:app --reload --log-level debug
```

### Model not loading

```bash
# Check if model files exist
ls models/cache/

# Retrain
python scripts/train_model.py

# Check model version in config
cat config/model_config.yaml
```

### Tests failing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run with verbose output
pytest -v -s

# Run a single test
pytest tests/test_health.py::test_health_endpoint -v
```

### Database issues

```bash
# SQLite reset (development)
rm mxmap.db
python scripts/seed_db.py

# PostgreSQL reset
docker-compose down -v
docker-compose up -d db
make seed
```

### Filtering endpoint errors

```bash
# Check EIS surrogate import
python3 -c "from app.ml.eis_surrogate import EISSurrogate; print('OK')"

# Verify geometry bounds (finger_width_um must be ≥ 2.0)
```

### Literature pipeline issues

```bash
# Ensure Ollama is running
ollama serve
ollama list   # Should show llama3

# Check pipeline logs
tail -f mxene-pipeline/data/logs/pipeline.log

# Test single PDF extraction
python3 -c "
from mxene_pipeline.src.extractor import MXeneExtractor
e = MXeneExtractor()
r = e.extract_from_pdf('path/to/paper.pdf')
print(r)
"
```

---

## 14. Development Reference

### Project File Map

```
app/
├── config.py               # pydantic-settings: all env vars
├── main.py                 # FastAPI(), CORS, routers, lifespan
├── core/dependencies.py    # get_db() → AsyncSession, get_predictor()
├── db/session.py           # create_engine() with SQLite/Postgres branching
├── ml/
│   ├── feature_engineer.py # fit_transform(), transform(), save/load
│   ├── xgboost_model.py    # train(), predict(), _evaluate_model()
│   ├── model_loader.py     # ModelLoader singleton, get_predictor()
│   ├── optimization.py     # MultiObjectiveOptimizer, Pareto filter
│   ├── eis_surrogate.py    # EISSurrogate, CPEParams, geometry_to_params()
│   └── printing_surrogates.py # PrintingSurrogate, InkWindow, FilmProperties
├── models/
│   ├── database.py         # Device, Prediction, TrainingMetadata, ...
│   └── schemas.py          # PredictionRequest, PredictionResponse, ...
└── api/v1/endpoints/
    ├── predictions.py      # predict(), predict_batch()
    ├── devices.py          # list_devices(), get_device(), create_device()
    ├── models.py           # get_model_metrics(), health_check()
    ├── advanced.py         # optimize(), explore(), compare(), recipes()
    ├── filtering.py        # predict_filtering(), optimize_filtering()
    └── printing.py         # recommend_printing(), estimate_printing()
```

### Key Constants

```python
# EIS surrogate defaults
BASE_CAPACITANCE_MF_CM2 = 15.0      # mF/cm² for MXene MSCs
DEFAULT_ALPHA_RANGE = (0.75, 0.95)  # CPE exponent bounds

# Optimization
POPULATION_SIZE = 60
GENERATIONS = 40
MAX_SOLUTIONS_RETURNED = 5

# Data decimation
MAX_PLOT_POINTS = 512

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.90
MEDIUM_CONFIDENCE_THRESHOLD = 0.70
```

### curl Examples

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Single prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"mxene_type":"Ti3C2Tx","terminations":"O","electrolyte":"H2SO4","thickness_um":5.0,"deposition_method":"vacuum_filtration"}'

# Batch prediction
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"devices":[{"mxene_type":"Ti3C2Tx","terminations":"O","electrolyte":"H2SO4","thickness_um":5.0,"deposition_method":"vacuum_filtration"},{"mxene_type":"Mo2CTx","terminations":"F","electrolyte":"KOH","thickness_um":10.0,"deposition_method":"spray_coating"}]}'

# List devices
curl "http://localhost:8000/api/v1/devices?page=1&page_size=10&mxene_type=Ti3C2Tx"

# Filtering predict
curl -X POST http://localhost:8000/api/v1/filtering/predict \
  -H "Content-Type: application/json" \
  -d '{"frequency_range_hz":[1,100000],"load_resistance_ohm":33.0,"geometry":{"finger_width_um":8,"finger_spacing_um":8,"finger_length_um":2000,"num_fingers_per_electrode":50,"overlap_length_um":1800,"thickness_nm":200,"substrate":"Si/SiO2"},"mxene_film":{"porosity_pct":30,"sheet_res_ohm_sq":10,"electrolyte":"PVA/H2SO4","process":"photolithography"}}'
```

### Dependencies

```toml
# Core
fastapi = ">=0.109"
sqlalchemy = {extras=["asyncio"], version=">=2.0"}
pydantic = ">=2.0"
pydantic-settings = ">=2.0"
aiosqlite = "*"
asyncpg = "*"

# ML
xgboost = ">=2.0"
scikit-learn = ">=1.4"
numpy = ">=1.26"
pandas = ">=2.1"
scipy = "*"

# Server
uvicorn = {extras=["standard"], version="*"}
jinja2 = "*"
python-multipart = "*"

# Dev & Test
pytest = "*"
pytest-asyncio = "*"
httpx = "*"
mypy = "*"
ruff = "*"
black = "*"
```

---

## Status

| Component | Status |
|-----------|--------|
| Core API (FastAPI) | ✅ Complete |
| Database schema + migrations | ✅ Complete |
| Synthetic data generation | ✅ Complete (300 samples) |
| XGBoost models + quantile regression | ✅ Complete |
| Feature engineering pipeline | ✅ Complete |
| Uncertainty quantification (95% CI) | ✅ Complete |
| Model persistence + versioning | ✅ Complete |
| Multi-objective optimization (Pareto) | ✅ Complete |
| Chemistry space exploration (UMAP) | ✅ Complete |
| Candidate comparison | ✅ Complete |
| Recipe card export | ✅ Complete |
| WebSocket real-time predictions | ✅ Complete |
| AC-line filtering (EIS surrogate) | ✅ Complete |
| Printing process design surrogate | ✅ Complete |
| Electrochromic visualization | ✅ Complete |
| Web UI (all pages) | ✅ Complete |
| Test suite (76 tests passing) | ✅ Complete |
| Literature extraction pipeline | ✅ Complete |
| Docker deployment | ✅ Complete |

---

*Built with FastAPI, SQLAlchemy, XGBoost, and the Python scientific stack.*
