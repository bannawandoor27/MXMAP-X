"""Prediction endpoints for ML model inference."""

import time
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import (
    PredictionRequest,
    PredictionResult,
    BatchPredictionRequest,
    BatchPredictionResult,
)
from app.ml.dummy import get_predictor
from app.core.exceptions import PredictionError

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResult,
    status_code=status.HTTP_200_OK,
    summary="Predict supercapacitor performance",
    description="""
    Generate performance predictions for a single MXene supercapacitor device.
    
    Returns predicted values with 95% confidence intervals for:
    - Areal capacitance (mF/cm²)
    - Equivalent series resistance (Ω)
    - Rate capability (%)
    - Cycle life (cycles to 80% retention)
    
    **Example Request:**
    ```json
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
    """,
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "areal_capacitance": {
                            "value": 350.5,
                            "lower_ci": 320.0,
                            "upper_ci": 381.0,
                            "confidence": "high",
                        },
                        "esr": {
                            "value": 2.5,
                            "lower_ci": 2.1,
                            "upper_ci": 2.9,
                            "confidence": "medium",
                        },
                        "rate_capability": {
                            "value": 85.0,
                            "lower_ci": 80.0,
                            "upper_ci": 90.0,
                            "confidence": "medium",
                        },
                        "cycle_life": {
                            "value": 10000,
                            "lower_ci": 8500,
                            "upper_ci": 11500,
                            "confidence": "medium",
                        },
                        "overall_confidence": "high",
                        "confidence_score": 0.92,
                        "model_version": "v0.1.0-dummy",
                        "prediction_time_ms": 15.3,
                        "request_id": "req_abc123",
                    }
                }
            },
        },
        422: {"description": "Validation error"},
        500: {"description": "Prediction error"},
    },
)
async def predict_single(request: PredictionRequest) -> PredictionResult:
    """
    Generate prediction for a single device.
    
    Args:
        request: Device composition and parameters
        
    Returns:
        Prediction result with uncertainty quantification
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        predictor = get_predictor()
        result = await predictor.predict(request)
        return result
    except PredictionError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": e.message,
                "error_code": e.error_code,
                "details": e.details,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": f"Unexpected error: {str(e)}",
                "error_code": "INTERNAL_ERROR",
            },
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResult,
    status_code=status.HTTP_200_OK,
    summary="Batch predict supercapacitor performance",
    description="""
    Generate performance predictions for multiple MXene supercapacitor devices.
    
    Maximum batch size: 100 devices
    
    **Example Request:**
    ```json
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
    """,
    responses={
        200: {"description": "Successful batch prediction"},
        422: {"description": "Validation error (e.g., batch too large)"},
        500: {"description": "Prediction error"},
    },
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResult:
    """
    Generate predictions for multiple devices.
    
    Args:
        request: Batch of device compositions
        
    Returns:
        Batch prediction results with statistics
        
    Raises:
        HTTPException: If batch prediction fails
    """
    start_time = time.time()
    
    try:
        predictor = get_predictor()
        predictions = await predictor.predict_batch(request.devices)
        
        total_time_ms = (time.time() - start_time) * 1000
        
        return BatchPredictionResult(
            predictions=predictions,
            total_count=len(request.devices),
            successful_count=len(predictions),
            failed_count=0,
            total_time_ms=total_time_ms,
        )
    except PredictionError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": e.message,
                "error_code": e.error_code,
                "details": e.details,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": f"Unexpected error: {str(e)}",
                "error_code": "INTERNAL_ERROR",
            },
        )
