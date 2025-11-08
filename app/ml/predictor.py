"""Abstract base class for ML predictors."""

from abc import ABC, abstractmethod
from typing import Any
from app.models.schemas import PredictionRequest, PredictionResult


class BasePredictor(ABC):
    """
    Abstract base class for ML prediction models.
    
    All predictor implementations must inherit from this class and implement
    the predict method.
    """

    def __init__(self, model_version: str, config: dict[str, Any]) -> None:
        """
        Initialize predictor.
        
        Args:
            model_version: Version identifier for the model
            config: Model configuration parameters
        """
        self.model_version = model_version
        self.config = config

    @abstractmethod
    async def predict(self, request: PredictionRequest) -> PredictionResult:
        """
        Generate prediction for a single device.
        
        Args:
            request: Device composition and parameters
            
        Returns:
            Prediction result with uncertainty quantification
            
        Raises:
            PredictionError: If prediction fails
        """
        pass

    @abstractmethod
    async def predict_batch(
        self, requests: list[PredictionRequest]
    ) -> list[PredictionResult]:
        """
        Generate predictions for multiple devices.
        
        Args:
            requests: List of device compositions
            
        Returns:
            List of prediction results
            
        Raises:
            PredictionError: If batch prediction fails
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """
        Get model metadata and configuration.
        
        Returns:
            Dictionary with model information
        """
        pass
