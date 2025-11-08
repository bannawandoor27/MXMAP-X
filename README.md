# MXMAP-X Backend API

AI-powered MXene supercapacitor design tool for predicting device performance metrics with uncertainty quantification.

## 🎯 Overview

MXMAP-X predicts supercapacitor performance from material composition and processing parameters:

- **Areal Capacitance** (mF/cm²)
- **Equivalent Series Resistance** (Ω)
- **Rate Capability** (%)
- **Cycle Life** (cycles to 80% retention)

All predictions include 95% confidence intervals and uncertainty quantification.

## 🏗️ Architecture

```
FastAPI + SQLAlchemy + PostgreSQL + scikit-learn/XGBoost
```

### Tech Stack

- **Framework**: FastAPI 0.109+
- **Database**: PostgreSQL 15+ with SQLAlchemy 2.0 (async)
- **Validation**: Pydantic v2
- **ML**: scikit-learn + XGBoost (currently dummy predictor)
- **Migrations**: Alembic
- **Dependency Management**: Poetry

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 15+
- Poetry

### 1. Clone and Install

```bash
# Clone repository
git clone <repository-url>
cd mxmap-x-backend

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your database credentials
# DATABASE_URL=postgresql://mxmap_user:mxmap_password@localhost:5432/mxmap_db
```

### 3. Start Database (Docker)

```bash
# Start PostgreSQL
docker-compose up -d db

# Or use local PostgreSQL and create database
createdb mxmap_db
```

### 4. Generate Synthetic Data

```bash
# Generate 300 physics-informed training samples
python scripts/generate_synthetic_data.py
```

### 5. Seed Database

```bash
# Run migrations and seed data
python scripts/seed_db.py
```

### 6. Start API Server

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 7. Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 📡 API Endpoints

### Health Check

```bash
GET /api/v1/health
```

### Single Prediction

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
  "areal_capacitance": {
    "value": 350.5,
    "lower_ci": 320.0,
    "upper_ci": 381.0,
    "confidence": "high"
  },
  "esr": {
    "value": 2.5,
    "lower_ci": 2.1,
    "upper_ci": 2.9,
    "confidence": "medium"
  },
  "rate_capability": {
    "value": 85.0,
    "lower_ci": 80.0,
    "upper_ci": 90.0,
    "confidence": "medium"
  },
  "cycle_life": {
    "value": 10000,
    "lower_ci": 8500,
    "upper_ci": 11500,
    "confidence": "medium"
  },
  "overall_confidence": "high",
  "confidence_score": 0.92,
  "model_version": "v0.1.0-dummy",
  "prediction_time_ms": 15.3,
  "request_id": "req_abc123"
}
```

### Batch Prediction

```bash
POST /api/v1/predict/batch
Content-Type: application/json

{
  "devices": [
    {
      "mxene_type": "Ti3C2Tx",
      "terminations": "O",
      "electrolyte": "H2SO4",
      "thickness_um": 5.0,
      "deposition_method": "vacuum_filtration"
    },
    {
      "mxene_type": "Mo2CTx",
      "terminations": "F",
      "electrolyte": "KOH",
      "thickness_um": 10.0,
      "deposition_method": "spray_coating"
    }
  ]
}
```

### List Training Devices

```bash
GET /api/v1/devices?page=1&page_size=50&mxene_type=Ti3C2Tx
```

### Get Device Details

```bash
GET /api/v1/devices/{device_id}
```

### Add Training Data

```bash
POST /api/v1/devices
Content-Type: application/json

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

### Model Metrics

```bash
GET /api/v1/models/metrics
```

## 🧪 Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=app --cov-report=html

# Type checking
mypy app

# Linting
ruff check app
black --check app
```

## 🗄️ Database Schema

### Device Table

Stores training data with material composition and performance metrics.

**Key Fields:**
- Material: `mxene_type`, `terminations`, `electrolyte`
- Processing: `thickness_um`, `deposition_method`, `annealing_temp_c`
- Structure: `interlayer_spacing_nm`, `specific_surface_area_m2g`
- Performance: `areal_capacitance_mf_cm2`, `esr_ohm`, `rate_capability_percent`, `cycle_life_cycles`

### Prediction Table

Stores ML predictions with uncertainty intervals.

### TrainingMetadata Table

Tracks model versions and performance metrics.

## 🔧 Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/mxmap_db
DATABASE_ECHO=false

# API
API_V1_PREFIX=/api/v1
PROJECT_NAME=MXMAP-X Backend
DEBUG=true

# CORS
BACKEND_CORS_ORIGINS=["http://localhost:3000"]

# ML Model
MODEL_CONFIG_PATH=config/model_config.yaml
MODEL_CACHE_DIR=models/cache
```

### Model Configuration

Edit `config/model_config.yaml` to configure:
- Feature selection
- Hyperparameters
- Uncertainty quantification
- Performance thresholds

## 📊 Synthetic Data

The `generate_synthetic_data.py` script creates 300 physics-informed samples with:

**Correlations:**
- Thicker films → higher capacitance, worse rate capability
- H₂SO₄ → higher capacitance, lower cycle life
- Ionic liquids → better high-temp performance, higher ESR
- Larger interlayer spacing → better rate capability

**Realistic Features:**
- 10-15% missing optical properties
- ±10-15% measurement noise
- Physically plausible ranges

## 🐳 Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## 🔄 Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## 📈 Next Steps (Phase 2)

1. **Real ML Models**
   - Train XGBoost models on synthetic data
   - Implement bootstrap uncertainty quantification
   - Add feature importance analysis

2. **Multi-Objective Optimization**
   - Pareto frontier calculation
   - Design space exploration
   - Trade-off visualization

3. **Advanced Features**
   - Active learning for data acquisition
   - Transfer learning from literature data
   - Explainable AI (SHAP values)

4. **Production Readiness**
   - API authentication (JWT)
   - Rate limiting
   - Caching (Redis)
   - Monitoring (Prometheus + Grafana)

## 📝 API Design Principles

- **Type Safety**: Strict Pydantic validation with enums
- **Error Handling**: Structured error responses with codes
- **Request Tracking**: Unique request IDs for debugging
- **Documentation**: Comprehensive OpenAPI specs with examples
- **Performance**: Async/await throughout, connection pooling

## 🤝 Contributing

This is a PhD application portfolio project. For questions or collaboration:

- Review API documentation at `/docs`
- Check database schema in `app/models/database.py`
- See Pydantic schemas in `app/models/schemas.py`

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

Built with FastAPI, SQLAlchemy, and the Python scientific stack.

---

**Status**: Phase 1 Complete ✅
- Core API with dummy predictor
- Database schema and migrations
- Synthetic data generation
- Comprehensive documentation

**Next**: Train real ML models and implement optimization features
