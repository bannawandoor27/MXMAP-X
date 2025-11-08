"""Tests for XGBoost model and feature engineering."""

import pytest
import numpy as np
import pandas as pd
from app.ml.feature_engineering import FeatureEngineer
from app.ml.xgboost_model import XGBoostPredictor
from app.models.schemas import PredictionRequest


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample training data."""
    return pd.DataFrame({
        "mxene_type": ["Ti3C2Tx", "Mo2CTx", "V2CTx"] * 10,
        "terminations": ["O", "OH", "F"] * 10,
        "electrolyte": ["H2SO4", "KOH", "ionic_liquid"] * 10,
        "electrolyte_concentration": [1.0, 2.0, None] * 10,
        "thickness_um": np.random.uniform(2, 10, 30),
        "deposition_method": ["vacuum_filtration", "spray_coating", "drop_casting"] * 10,
        "annealing_temp_c": [120.0, None, 150.0] * 10,
        "annealing_time_min": [60.0, None, 90.0] * 10,
        "interlayer_spacing_nm": np.random.uniform(1.0, 1.5, 30),
        "specific_surface_area_m2g": np.random.uniform(50, 150, 30),
        "pore_volume_cm3g": np.random.uniform(0.05, 0.25, 30),
        "optical_transmittance": [75.0, None, 80.0] * 10,
        "sheet_resistance_ohm_sq": [50.0, 60.0, None] * 10,
    })


@pytest.fixture
def sample_targets() -> dict[str, np.ndarray]:
    """Create sample target values."""
    return {
        "capacitance": np.random.uniform(200, 500, 30),
        "esr": np.random.uniform(1, 5, 30),
        "rate_capability": np.random.uniform(70, 95, 30),
        "cycle_life": np.random.randint(5000, 15000, 30),
    }


def test_feature_engineer_fit_transform(sample_data: pd.DataFrame) -> None:
    """Test feature engineering pipeline."""
    engineer = FeatureEngineer()
    
    # Fit and transform
    X = engineer.fit_transform(sample_data)
    
    # Check output shape
    assert X.shape[0] == len(sample_data)
    assert X.shape[1] > 0
    
    # Check feature names
    feature_names = engineer.get_feature_names()
    assert len(feature_names) == X.shape[1]
    assert "mxene_type" in feature_names
    assert "thickness_um" in feature_names
    assert "thickness_squared" in feature_names


def test_feature_engineer_handles_missing_values(sample_data: pd.DataFrame) -> None:
    """Test that feature engineer handles missing values."""
    engineer = FeatureEngineer()
    
    # Transform with missing values
    X = engineer.fit_transform(sample_data)
    
    # Check no NaN values in output
    assert not np.isnan(X).any()


def test_feature_engineer_persistence(sample_data: pd.DataFrame, tmp_path) -> None:
    """Test saving and loading feature engineer."""
    engineer = FeatureEngineer()
    engineer.fit(sample_data)
    
    # Save
    save_path = tmp_path / "feature_engineer.joblib"
    engineer.save(str(save_path))
    
    # Load
    loaded_engineer = FeatureEngineer.load(str(save_path))
    
    # Check same transformation
    X1 = engineer.transform(sample_data)
    X2 = loaded_engineer.transform(sample_data)
    
    np.testing.assert_array_almost_equal(X1, X2)


def test_xgboost_predictor_training(
    sample_data: pd.DataFrame,
    sample_targets: dict[str, np.ndarray],
) -> None:
    """Test XGBoost model training."""
    config = {
        "hyperparameters": {
            "xgboost": {
                "n_estimators": 10,  # Small for testing
                "max_depth": 3,
                "learning_rate": 0.1,
            }
        }
    }
    
    predictor = XGBoostPredictor(
        model_version="test_v1",
        config=config,
    )
    
    # Train
    results = predictor.train(sample_data, sample_targets)
    
    # Check models were trained
    assert "capacitance" in predictor.models
    assert "mean" in predictor.models["capacitance"]
    assert "lower" in predictor.models["capacitance"]
    assert "upper" in predictor.models["capacitance"]
    
    # Check metrics
    assert len(predictor.metrics) > 0


@pytest.mark.asyncio
async def test_xgboost_predictor_prediction(
    sample_data: pd.DataFrame,
    sample_targets: dict[str, np.ndarray],
) -> None:
    """Test XGBoost model prediction."""
    config = {
        "hyperparameters": {
            "xgboost": {
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
            }
        }
    }
    
    predictor = XGBoostPredictor(
        model_version="test_v1",
        config=config,
    )
    
    # Train
    predictor.train(sample_data, sample_targets)
    
    # Create prediction request
    request = PredictionRequest(
        mxene_type="Ti3C2Tx",
        terminations="O",
        electrolyte="H2SO4",
        electrolyte_concentration=1.0,
        thickness_um=5.0,
        deposition_method="vacuum_filtration",
        annealing_temp_c=120.0,
        annealing_time_min=60.0,
        interlayer_spacing_nm=1.2,
        specific_surface_area_m2g=98.5,
    )
    
    # Generate prediction
    result = await predictor.predict(request)
    
    # Check result structure
    assert result.areal_capacitance.value > 0
    assert result.areal_capacitance.lower_ci < result.areal_capacitance.value
    assert result.areal_capacitance.upper_ci > result.areal_capacitance.value
    assert result.overall_confidence in ["high", "medium", "low"]
    assert 0 <= result.confidence_score <= 1


@pytest.mark.asyncio
async def test_xgboost_predictor_batch_prediction(
    sample_data: pd.DataFrame,
    sample_targets: dict[str, np.ndarray],
) -> None:
    """Test XGBoost batch prediction."""
    config = {
        "hyperparameters": {
            "xgboost": {
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
            }
        }
    }
    
    predictor = XGBoostPredictor(
        model_version="test_v1",
        config=config,
    )
    
    # Train
    predictor.train(sample_data, sample_targets)
    
    # Create batch requests
    requests = [
        PredictionRequest(
            mxene_type="Ti3C2Tx",
            terminations="O",
            electrolyte="H2SO4",
            thickness_um=5.0,
            deposition_method="vacuum_filtration",
        ),
        PredictionRequest(
            mxene_type="Mo2CTx",
            terminations="F",
            electrolyte="KOH",
            thickness_um=10.0,
            deposition_method="spray_coating",
        ),
    ]
    
    # Generate predictions
    results = await predictor.predict_batch(requests)
    
    # Check results
    assert len(results) == 2
    assert all(r.areal_capacitance.value > 0 for r in results)


def test_xgboost_predictor_persistence(
    sample_data: pd.DataFrame,
    sample_targets: dict[str, np.ndarray],
    tmp_path,
) -> None:
    """Test saving and loading XGBoost predictor."""
    config = {
        "hyperparameters": {
            "xgboost": {
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
            }
        }
    }
    
    predictor = XGBoostPredictor(
        model_version="test_v1",
        config=config,
    )
    
    # Train
    predictor.train(sample_data, sample_targets)
    
    # Save
    save_path = tmp_path / "model"
    predictor.save(str(save_path))
    
    # Load
    loaded_predictor = XGBoostPredictor.load(str(save_path))
    
    # Check same model version
    assert loaded_predictor.model_version == predictor.model_version
    
    # Check metrics preserved
    assert len(loaded_predictor.metrics) == len(predictor.metrics)


def test_feature_importance(
    sample_data: pd.DataFrame,
    sample_targets: dict[str, np.ndarray],
) -> None:
    """Test feature importance calculation."""
    config = {
        "hyperparameters": {
            "xgboost": {
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
            }
        }
    }
    
    predictor = XGBoostPredictor(
        model_version="test_v1",
        config=config,
    )
    
    # Train
    predictor.train(sample_data, sample_targets)
    
    # Get feature importance
    importance = predictor.get_feature_importance()
    
    # Check structure
    assert isinstance(importance, dict)
    assert len(importance) > 0
    
    # Check values sum to 1 (normalized)
    assert abs(sum(importance.values()) - 1.0) < 0.01
    
    # Check all values are non-negative
    assert all(v >= 0 for v in importance.values())
