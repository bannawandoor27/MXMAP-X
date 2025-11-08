"""Model loader for managing trained models."""

import os
from pathlib import Path
from typing import Union

from app.ml.predictor import BasePredictor
from app.ml.dummy import DummyPredictor
from app.ml.xgboost_model import XGBoostPredictor
from app.config import settings


class ModelLoader:
    """
    Singleton model loader for managing ML models.
    
    Automatically loads trained XGBoost model if available,
    otherwise falls back to dummy predictor.
    """

    _instance: "ModelLoader | None" = None
    _predictor: BasePredictor | None = None

    def __new__(cls) -> "ModelLoader":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_predictor(self) -> BasePredictor:
        """
        Get active predictor instance.
        
        Returns:
            Active predictor (XGBoost or Dummy)
        """
        if self._predictor is None:
            self._predictor = self._load_predictor()
        return self._predictor

    def _load_predictor(self) -> BasePredictor:
        """
        Load predictor from disk or create dummy.
        
        Returns:
            Loaded or dummy predictor
        """
        model_dir = settings.MODEL_CACHE_DIR
        
        # Check if trained model exists
        if self._trained_model_exists(model_dir):
            try:
                print(f"Loading trained XGBoost model from {model_dir}...")
                predictor = XGBoostPredictor.load(model_dir)
                print(f"✓ Loaded model version: {predictor.model_version}")
                return predictor
            except Exception as e:
                print(f"⚠ Failed to load trained model: {e}")
                print("Falling back to dummy predictor")
        else:
            print("No trained model found, using dummy predictor")
        
        # Fall back to dummy predictor
        return DummyPredictor()

    def _trained_model_exists(self, model_dir: str) -> bool:
        """
        Check if trained model files exist.
        
        Args:
            model_dir: Model directory path
            
        Returns:
            True if model files exist
        """
        required_files = [
            "metadata.joblib",
            "feature_engineer.joblib",
            "capacitance_mean.json",
            "esr_mean.json",
            "rate_capability_mean.json",
            "cycle_life_mean.json",
        ]
        
        return all(
            os.path.exists(os.path.join(model_dir, f))
            for f in required_files
        )

    def reload_model(self) -> BasePredictor:
        """
        Reload model from disk.
        
        Useful after training a new model.
        
        Returns:
            Reloaded predictor
        """
        self._predictor = None
        return self.get_predictor()

    def get_model_info(self) -> dict[str, any]:
        """
        Get information about active model.
        
        Returns:
            Model metadata
        """
        predictor = self.get_predictor()
        return predictor.get_model_info()


# Global model loader instance
_model_loader: ModelLoader | None = None


def get_model_loader() -> ModelLoader:
    """Get global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader


def get_predictor() -> BasePredictor:
    """
    Get active predictor instance.
    
    This is the main function to use throughout the application.
    
    Returns:
        Active predictor
    """
    loader = get_model_loader()
    return loader.get_predictor()
