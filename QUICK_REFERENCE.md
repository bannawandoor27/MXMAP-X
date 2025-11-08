# MXMAP-X Quick Reference

Fast reference for common tasks and commands.

## 🚀 Setup (First Time)

```bash
# Install dependencies
poetry install

# Start database
docker-compose up -d db

# Generate data and train models
make ml-pipeline

# Start API
make run
```

## 🔄 Common Commands

### Development

```bash
make run          # Start API server (with reload)
make test         # Run all tests
make lint         # Run linters (mypy, ruff)
make format       # Format code (black, ruff)
```

### Machine Learning

```bash
make train        # Train XGBoost models
make evaluate     # Evaluate model performance
make ml-pipeline  # Full pipeline (seed + train + evaluate)
```

### Database

```bash
make seed         # Generate data and seed database
make migrate      # Run database migrations
make docker-up    # Start all Docker services
make docker-down  # Stop all Docker services
```

## 📡 API Endpoints

### Health & Info

```bash
GET  /api/v1/health              # Health check
GET  /api/v1/models/metrics      # Model performance
GET  /docs                       # Swagger UI
```

### Predictions

```bash
POST /api/v1/predict             # Single prediction
POST /api/v1/predict/batch       # Batch predictions (max 100)
```

### Devices

```bash
GET  /api/v1/devices             # List devices (paginated)
GET  /api/v1/devices/{id}        # Get device details
POST /api/v1/devices             # Add training data
```

### Advanced Features

```bash
POST /api/v1/optimize            # Multi-objective optimization
GET  /api/v1/explore             # Chemistry space map (UMAP)
POST /api/v1/compare             # Compare candidates
GET  /api/v1/recipes/{id}        # Export recipe card
WS   /api/v1/ws/predict          # WebSocket predictions
```

## 🧪 Example Requests

### Single Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "mxene_type": "Ti3C2Tx",
    "terminations": "O",
    "electrolyte": "H2SO4",
    "thickness_um": 5.0,
    "deposition_method": "vacuum_filtration"
  }'
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "devices": [
      {"mxene_type": "Ti3C2Tx", "terminations": "O", "electrolyte": "H2SO4", "thickness_um": 5.0, "deposition_method": "vacuum_filtration"},
      {"mxene_type": "Mo2CTx", "terminations": "F", "electrolyte": "KOH", "thickness_um": 10.0, "deposition_method": "spray_coating"}
    ]
  }'
```

### List Devices

```bash
curl "http://localhost:8000/api/v1/devices?page=1&page_size=10&mxene_type=Ti3C2Tx"
```

## 🐍 Python Usage

### Make Prediction

```python
import asyncio
from app.ml.model_loader import get_predictor
from app.models.schemas import PredictionRequest

async def predict():
    predictor = get_predictor()
    
    request = PredictionRequest(
        mxene_type="Ti3C2Tx",
        terminations="O",
        electrolyte="H2SO4",
        thickness_um=5.0,
        deposition_method="vacuum_filtration",
    )
    
    result = await predictor.predict(request)
    print(f"Capacitance: {result.areal_capacitance.value} mF/cm²")
    print(f"Confidence: {result.overall_confidence}")

asyncio.run(predict())
```

### Train Model

```python
from scripts.train_model import ModelTrainer
import asyncio

async def train():
    trainer = ModelTrainer()
    df = trainer.load_data()
    X_train, y_train, X_test, y_test = trainer.prepare_data(df)
    predictor = trainer.train_model(X_train, y_train, X_test, y_test)
    trainer.save_model(predictor)
    await trainer.save_metadata_to_db(predictor)

asyncio.run(train())
```

## 📊 Model Performance

| Target | R² | RMSE | Coverage |
|--------|-----|------|----------|
| Capacitance | 0.952 | 24.3 mF/cm² | 94.2% |
| ESR | 0.888 | 0.34 Ω | 93.8% |
| Rate Capability | 0.823 | 4.9% | 95.5% |
| Cycle Life | 0.795 | 1824 cycles | 92.1% |

## 🔧 Configuration Files

- `.env` - Environment variables
- `config/model_config.yaml` - ML hyperparameters
- `alembic.ini` - Database migrations
- `pyproject.toml` - Dependencies

## 📁 Important Directories

- `app/` - Application code
- `app/ml/` - ML models and pipelines
- `scripts/` - Training and utility scripts
- `tests/` - Test suite
- `docs/` - Documentation
- `models/cache/` - Trained models (gitignored)
- `reports/` - Evaluation reports (gitignored)
- `data/` - Training data (gitignored)

## 🐛 Troubleshooting

### API won't start
```bash
# Check database connection
docker-compose ps
# Check logs
docker-compose logs api
```

### Model not loading
```bash
# Check if model files exist
ls models/cache/
# Retrain if needed
make train
```

### Tests failing
```bash
# Install test dependencies
poetry install
# Run with verbose output
pytest -v -s
```

### Database issues
```bash
# Reset database
docker-compose down -v
docker-compose up -d db
make seed
```

## 📚 Documentation

- [README.md](README.md) - Main documentation
- [docs/API_EXAMPLES.md](docs/API_EXAMPLES.md) - API usage examples
- [docs/ML_PIPELINE.md](docs/ML_PIPELINE.md) - ML pipeline details
- [docs/IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md) - Implementation overview

## 🔗 URLs

- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/v1/health

## 📦 Dependencies

### Core
- FastAPI 0.109+
- SQLAlchemy 2.0+
- Pydantic v2
- PostgreSQL 15+

### ML
- XGBoost 2.0+
- scikit-learn 1.4+
- NumPy 1.26+
- pandas 2.1+

### Dev
- pytest
- mypy
- black
- ruff

## 🎯 Quick Checks

```bash
# Is API running?
curl http://localhost:8000/api/v1/health

# Is database connected?
docker-compose ps

# Are models trained?
ls models/cache/

# Run tests
pytest

# Check code quality
make lint
```

---

**For detailed information, see the full documentation in the `docs/` directory.**
