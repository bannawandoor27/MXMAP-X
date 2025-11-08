"""XGBoost model with quantile regression for uncertainty quantification."""

import time
import uuid
from typing import Any
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

from app.ml.predictor import BasePredictor
from app.ml.feature_engineering import FeatureEngineer
from app.models.schemas import (
    PredictionRequest,
    PredictionResult,
    UncertaintyInterval,
    ConfidenceLevel,
)
from app.core.exceptions import PredictionError


class XGBoostPredictor(BasePredictor):
    """
    XGBoost predictor with quantile regression for uncertainty quantification.
    
    Uses three models per target:
    - Main model (mean prediction)
    - Lower quantile model (5th percentile)
    - Upper quantile model (95th percentile)
    """

    def __init__(
        self,
        model_version: str,
        config: dict[str, Any],
        feature_engineer: FeatureEngineer | None = None,
    ) -> None:
        """
        Initialize XGBoost predictor.
        
        Args:
            model_version: Model version identifier
            config: Model configuration
            feature_engineer: Fitted feature engineer
        """
        super().__init__(model_version, config)
        
        self.feature_engineer = feature_engineer or FeatureEngineer()
        
        # Models for each target (mean, lower, upper quantiles)
        self.models: dict[str, dict[str, xgb.XGBRegressor]] = {
            "capacitance": {},
            "esr": {},
            "rate_capability": {},
            "cycle_life": {},
        }
        
        # Performance metrics
        self.metrics: dict[str, dict[str, float]] = {}
        
        # Feature importance
        self.feature_importance: dict[str, float] = {}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: dict[str, np.ndarray],
        X_val: pd.DataFrame | None = None,
        y_val: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """
        Train XGBoost models with quantile regression.
        
        Args:
            X_train: Training features
            y_train: Training targets (dict with keys: capacitance, esr, rate_capability, cycle_life)
            X_val: Optional validation features
            y_val: Optional validation targets
            
        Returns:
            Training metrics
        """
        # Fit feature engineer
        self.feature_engineer.fit(X_train)
        
        # Transform features
        X_train_transformed = self.feature_engineer.transform(X_train)
        X_val_transformed = None
        if X_val is not None:
            X_val_transformed = self.feature_engineer.transform(X_val)
        
        # Train models for each target
        training_results = {}
        
        for target_name, target_values in y_train.items():
            print(f"\nTraining models for {target_name}...")
            
            # Train mean model
            mean_model = self._train_single_model(
                X_train_transformed,
                target_values,
                objective="reg:squarederror",
                quantile=None,
            )
            self.models[target_name]["mean"] = mean_model
            
            # Train lower quantile model (5th percentile)
            lower_model = self._train_single_model(
                X_train_transformed,
                target_values,
                objective="reg:quantileerror",
                quantile=0.05,
            )
            self.models[target_name]["lower"] = lower_model
            
            # Train upper quantile model (95th percentile)
            upper_model = self._train_single_model(
                X_train_transformed,
                target_values,
                objective="reg:quantileerror",
                quantile=0.95,
            )
            self.models[target_name]["upper"] = upper_model
            
            # Evaluate on validation set
            if X_val_transformed is not None and y_val is not None:
                val_metrics = self._evaluate_model(
                    mean_model,
                    X_val_transformed,
                    y_val[target_name],
                )
                self.metrics[target_name] = val_metrics
                training_results[target_name] = val_metrics
                
                print(f"  R²: {val_metrics['r2']:.4f}")
                print(f"  RMSE: {val_metrics['rmse']:.4f}")
                print(f"  MAE: {val_metrics['mae']:.4f}")
        
        # Calculate feature importance (average across all mean models)
        self._calculate_feature_importance()
        
        return training_results

    def _train_single_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        objective: str,
        quantile: float | None,
    ) -> xgb.XGBRegressor:
        """Train a single XGBoost model."""
        params = self.config.get("hyperparameters", {}).get("xgboost", {})
        
        # Create model
        if objective == "reg:quantileerror" and quantile is not None:
            model = xgb.XGBRegressor(
                objective=objective,
                quantile_alpha=quantile,
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.05),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                min_child_weight=params.get("min_child_weight", 3),
                gamma=params.get("gamma", 0.1),
                reg_alpha=params.get("reg_alpha", 0.1),
                reg_lambda=params.get("reg_lambda", 1.0),
                random_state=params.get("random_state", 42),
                n_jobs=-1,
            )
        else:
            model = xgb.XGBRegressor(
                objective=objective,
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.05),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                min_child_weight=params.get("min_child_weight", 3),
                gamma=params.get("gamma", 0.1),
                reg_alpha=params.get("reg_alpha", 0.1),
                reg_lambda=params.get("reg_lambda", 1.0),
                random_state=params.get("random_state", 42),
                n_jobs=-1,
            )
        
        # Train model
        model.fit(X, y, verbose=False)
        
        return model

    def _evaluate_model(
        self,
        model: xgb.XGBRegressor,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate model performance."""
        y_pred = model.predict(X)
        
        return {
            "r2": r2_score(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
        }

    def _calculate_feature_importance(self) -> None:
        """Calculate average feature importance across all mean models."""
        feature_names = self.feature_engineer.get_feature_names()
        importance_sum = np.zeros(len(feature_names))
        
        for target_name in self.models:
            if "mean" in self.models[target_name]:
                importance = self.models[target_name]["mean"].feature_importances_
                importance_sum += importance
        
        # Average and normalize
        importance_avg = importance_sum / len(self.models)
        importance_normalized = importance_avg / importance_avg.sum()
        
        self.feature_importance = {
            name: float(imp)
            for name, imp in zip(feature_names, importance_normalized)
        }

    async def predict(self, request: PredictionRequest) -> PredictionResult:
        """
        Generate prediction with uncertainty quantification.
        
        Args:
            request: Device composition and parameters
            
        Returns:
            Prediction result with confidence intervals
        """
        start_time = time.time()
        
        try:
            # Convert request to DataFrame
            df = pd.DataFrame([request.model_dump()])
            
            # Transform features
            X = self.feature_engineer.transform(df)
            
            # Generate predictions for each target
            capacitance = self._predict_with_uncertainty(X, "capacitance")
            esr = self._predict_with_uncertainty(X, "esr")
            rate_capability = self._predict_with_uncertainty(X, "rate_capability")
            cycle_life = self._predict_with_uncertainty(X, "cycle_life")
            
            # Calculate overall confidence
            confidence_score = self._calculate_confidence(request, X)
            overall_confidence = self._get_confidence_level(confidence_score)
            
            # Generate request ID
            request_id = f"req_{uuid.uuid4().hex[:12]}"
            
            prediction_time_ms = (time.time() - start_time) * 1000
            
            return PredictionResult(
                areal_capacitance=capacitance,
                esr=esr,
                rate_capability=rate_capability,
                cycle_life=cycle_life,
                overall_confidence=overall_confidence,
                confidence_score=confidence_score,
                model_version=self.model_version,
                prediction_time_ms=prediction_time_ms,
                request_id=request_id,
            )
            
        except Exception as e:
            raise PredictionError(
                message=f"Prediction failed: {str(e)}",
                details={"request": request.model_dump()},
            )

    async def predict_batch(
        self, requests: list[PredictionRequest]
    ) -> list[PredictionResult]:
        """Generate predictions for multiple devices."""
        results = []
        for request in requests:
            result = await self.predict(request)
            results.append(result)
        return results

    def _predict_with_uncertainty(
        self, X: np.ndarray, target_name: str
    ) -> UncertaintyInterval:
        """
        Predict with uncertainty quantification.
        
        Args:
            X: Feature matrix
            target_name: Target metric name
            
        Returns:
            Uncertainty interval with value and confidence bounds
        """
        # Get predictions from all three models
        mean_pred = self.models[target_name]["mean"].predict(X)[0]
        lower_pred = self.models[target_name]["lower"].predict(X)[0]
        upper_pred = self.models[target_name]["upper"].predict(X)[0]
        
        # Ensure proper ordering
        lower_ci = min(lower_pred, mean_pred)
        upper_ci = max(upper_pred, mean_pred)
        
        # Calculate confidence based on interval width
        interval_width = upper_ci - lower_ci
        relative_width = interval_width / (mean_pred + 1e-6)
        
        if relative_width < 0.15:
            confidence = ConfidenceLevel.HIGH
        elif relative_width < 0.30:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW
        
        # Round appropriately
        if target_name == "cycle_life":
            return UncertaintyInterval(
                value=int(round(mean_pred)),
                lower_ci=int(round(lower_ci)),
                upper_ci=int(round(upper_ci)),
                confidence=confidence,
            )
        else:
            decimals = 3 if target_name == "esr" else 2
            return UncertaintyInterval(
                value=round(float(mean_pred), decimals),
                lower_ci=round(float(lower_ci), decimals),
                upper_ci=round(float(upper_ci), decimals),
                confidence=confidence,
            )

    def _calculate_confidence(
        self, request: PredictionRequest, X: np.ndarray
    ) -> float:
        """
        Calculate overall confidence score.
        
        Based on:
        - Data completeness
        - Model performance metrics
        - Prediction uncertainty
        """
        confidence = 0.7  # Base confidence
        
        # Bonus for optional parameters
        if request.electrolyte_concentration is not None:
            confidence += 0.04
        if request.annealing_temp_c is not None:
            confidence += 0.04
        if request.interlayer_spacing_nm is not None:
            confidence += 0.04
        if request.specific_surface_area_m2g is not None:
            confidence += 0.04
        if request.pore_volume_cm3g is not None:
            confidence += 0.03
        if request.optical_transmittance is not None:
            confidence += 0.02
        if request.sheet_resistance_ohm_sq is not None:
            confidence += 0.02
        
        # Adjust based on model performance
        if self.metrics:
            avg_r2 = np.mean([m.get("r2", 0.8) for m in self.metrics.values()])
            confidence *= (0.8 + 0.2 * avg_r2)  # Scale by R²
        
        return float(np.clip(confidence, 0.0, 1.0))

    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert numerical confidence to categorical level."""
        if score >= 0.9:
            return ConfidenceLevel.HIGH
        elif score >= 0.7:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance

    def get_model_info(self) -> dict[str, Any]:
        """Get model metadata."""
        return {
            "model_version": self.model_version,
            "model_type": "xgboost",
            "description": "XGBoost with quantile regression for uncertainty",
            "features": self.feature_engineer.get_feature_names(),
            "targets": list(self.models.keys()),
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
        }

    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Directory path to save model files
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save feature engineer
        self.feature_engineer.save(f"{path}/feature_engineer.joblib")
        
        # Save models
        for target_name, models in self.models.items():
            for model_type, model in models.items():
                model.save_model(f"{path}/{target_name}_{model_type}.json")
        
        # Save metadata
        metadata = {
            "model_version": self.model_version,
            "config": self.config,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
        }
        joblib.dump(metadata, f"{path}/metadata.joblib")

    @classmethod
    def load(cls, path: str) -> "XGBoostPredictor":
        """
        Load model from disk.
        
        Args:
            path: Directory path containing model files
            
        Returns:
            Loaded XGBoost predictor
        """
        # Load metadata
        metadata = joblib.load(f"{path}/metadata.joblib")
        
        # Load feature engineer
        feature_engineer = FeatureEngineer.load(f"{path}/feature_engineer.joblib")
        
        # Create predictor instance
        predictor = cls(
            model_version=metadata["model_version"],
            config=metadata["config"],
            feature_engineer=feature_engineer,
        )
        
        # Load models
        targets = ["capacitance", "esr", "rate_capability", "cycle_life"]
        model_types = ["mean", "lower", "upper"]
        
        for target_name in targets:
            for model_type in model_types:
                model_path = f"{path}/{target_name}_{model_type}.json"
                model = xgb.XGBRegressor()
                model.load_model(model_path)
                predictor.models[target_name][model_type] = model
        
        # Restore metrics and feature importance
        predictor.metrics = metadata["metrics"]
        predictor.feature_importance = metadata["feature_importance"]
        
        return predictor
