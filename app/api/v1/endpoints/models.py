"""Model metadata and performance endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_db
from app.models.database import TrainingMetadata
from app.models.schemas import ModelMetrics, HealthCheckResponse
from app.ml.model_loader import get_predictor
from app.config import settings
from app.db.session import engine

router = APIRouter()


@router.get(
    "/models/metrics",
    response_model=ModelMetrics,
    status_code=status.HTTP_200_OK,
    summary="Get model performance metrics",
    description="""
    Retrieve performance metrics for the active ML model.
    
    Returns:
    - Model version and type
    - R² scores for each target metric
    - Training/test dataset sizes
    - Training timestamp
    
    **Example Response:**
    ```json
    {
        "model_version": "v0.1.0-dummy",
        "model_type": "dummy",
        "metrics": {
            "capacitance_r2": 0.95,
            "esr_r2": 0.88,
            "rate_capability_r2": 0.82,
            "cycle_life_r2": 0.79
        },
        "training_samples": 300,
        "test_samples": 75,
        "trained_at": "2024-01-15T10:00:00",
        "is_active": true
    }
    ```
    """,
)
async def get_model_metrics(
    db: AsyncSession = Depends(get_db),
) -> ModelMetrics:
    """
    Get active model performance metrics.
    
    Args:
        db: Database session
        
    Returns:
        Model metrics and metadata
    """
    try:
        # Try to get active model from database
        query = select(TrainingMetadata).where(TrainingMetadata.is_active == 1)
        result = await db.execute(query)
        model_metadata = result.scalar_one_or_none()
        
        if model_metadata:
            # Return database metrics
            return ModelMetrics(
                model_version=model_metadata.model_version,
                model_type=model_metadata.model_type,
                metrics={
                    "capacitance_r2": model_metadata.test_r2_capacitance or 0.0,
                    "esr_r2": model_metadata.test_r2_esr or 0.0,
                    "rate_capability_r2": model_metadata.test_r2_rate_capability or 0.0,
                    "cycle_life_r2": model_metadata.test_r2_cycle_life or 0.0,
                    "capacitance_rmse": model_metadata.test_rmse_capacitance or 0.0,
                },
                training_samples=model_metadata.training_samples,
                test_samples=model_metadata.test_samples,
                trained_at=model_metadata.trained_at,
                is_active=True,
            )
        else:
            # Return dummy model metrics
            predictor = get_predictor()
            info = predictor.get_model_info()
            
            return ModelMetrics(
                model_version=predictor.model_version,
                model_type="dummy",
                metrics={
                    "capacitance_r2": 0.95,
                    "esr_r2": 0.88,
                    "rate_capability_r2": 0.82,
                    "cycle_life_r2": 0.79,
                },
                training_samples=300,
                test_samples=75,
                trained_at="2024-01-15T10:00:00",
                is_active=True,
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": f"Failed to retrieve model metrics: {str(e)}",
                "error_code": "DATABASE_ERROR",
            },
        )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="""
    Check API health status.
    
    Verifies:
    - API is running
    - Database connection
    - ML model loaded
    
    **Example Response:**
    ```json
    {
        "status": "healthy",
        "version": "0.1.0",
        "database": "connected",
        "model": "loaded"
    }
    ```
    """,
)
async def health_check(
    db: AsyncSession = Depends(get_db),
) -> HealthCheckResponse:
    """
    Perform health check.
    
    Args:
        db: Database session
        
    Returns:
        Health status
    """
    # Check database connection
    try:
        await db.execute(select(1))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"
    
    # Check model
    try:
        predictor = get_predictor()
        model_status = "loaded"
    except Exception:
        model_status = "not_loaded"
    
    # Overall status
    overall_status = "healthy" if db_status == "connected" and model_status == "loaded" else "unhealthy"
    
    return HealthCheckResponse(
        status=overall_status,
        version=settings.VERSION,
        database=db_status,
        model=model_status,
    )
