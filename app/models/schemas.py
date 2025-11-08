"""Pydantic schemas for request/response validation."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


# ============================================================================
# Enums for Validation
# ============================================================================


class MXeneType(str, Enum):
    """Supported MXene types."""

    TI3C2TX = "Ti3C2Tx"
    MO2CTX = "Mo2CTx"
    V2CTX = "V2CTx"
    TI2CTX = "Ti2CTx"
    NB2CTX = "Nb2CTx"
    TA4C3TX = "Ta4C3Tx"
    TI3CNTX = "Ti3CNTx"


class Termination(str, Enum):
    """Surface termination types."""

    O = "O"
    OH = "OH"
    F = "F"
    MIXED = "mixed"
    CL = "Cl"


class Electrolyte(str, Enum):
    """Electrolyte types."""

    H2SO4 = "H2SO4"
    KOH = "KOH"
    NAOH = "NaOH"
    IONIC_LIQUID = "ionic_liquid"
    EMIMBF4 = "EMIMBF4"
    PVA_H2SO4 = "PVA_H2SO4"
    PVA_KOH = "PVA_KOH"
    ORGANIC = "organic"


class DepositionMethod(str, Enum):
    """Deposition methods."""

    VACUUM_FILTRATION = "vacuum_filtration"
    SPRAY_COATING = "spray_coating"
    DROP_CASTING = "drop_casting"
    SPIN_COATING = "spin_coating"
    BLADE_COATING = "blade_coating"
    INKJET_PRINTING = "inkjet_printing"


class ConfidenceLevel(str, Enum):
    """Prediction confidence levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ============================================================================
# Base Schemas
# ============================================================================


class DeviceCompositionBase(BaseModel):
    """Base schema for device composition parameters."""

    mxene_type: MXeneType = Field(
        ...,
        description="MXene formula",
        examples=["Ti3C2Tx"],
    )
    terminations: Termination = Field(
        ...,
        description="Surface terminations",
        examples=["O"],
    )
    electrolyte: Electrolyte = Field(
        ...,
        description="Electrolyte type",
        examples=["H2SO4"],
    )
    electrolyte_concentration: Optional[float] = Field(
        None,
        ge=0.01,
        le=20.0,
        description="Electrolyte concentration in M",
        examples=[1.0],
    )
    thickness_um: float = Field(
        ...,
        ge=0.5,
        le=50.0,
        description="Film thickness in micrometers",
        examples=[5.0],
    )
    deposition_method: DepositionMethod = Field(
        ...,
        description="Deposition method",
        examples=["vacuum_filtration"],
    )
    annealing_temp_c: Optional[float] = Field(
        None,
        ge=25.0,
        le=500.0,
        description="Annealing temperature in Celsius",
        examples=[120.0],
    )
    annealing_time_min: Optional[float] = Field(
        None,
        ge=0.0,
        le=1440.0,
        description="Annealing time in minutes",
        examples=[60.0],
    )


class StructuralPropertiesBase(BaseModel):
    """Base schema for structural properties."""

    interlayer_spacing_nm: Optional[float] = Field(
        None,
        ge=0.5,
        le=5.0,
        description="Interlayer spacing in nanometers",
        examples=[1.2],
    )
    specific_surface_area_m2g: Optional[float] = Field(
        None,
        ge=1.0,
        le=500.0,
        description="Specific surface area in m²/g",
        examples=[98.5],
    )
    pore_volume_cm3g: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Pore volume in cm³/g",
        examples=[0.15],
    )


class OpticalPropertiesBase(BaseModel):
    """Base schema for optical properties (may be missing)."""

    optical_transmittance: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Optical transmittance percentage",
        examples=[75.0],
    )
    sheet_resistance_ohm_sq: Optional[float] = Field(
        None,
        ge=0.1,
        le=10000.0,
        description="Sheet resistance in Ω/sq",
        examples=[50.0],
    )


class PerformanceMetricsBase(BaseModel):
    """Base schema for performance metrics."""

    areal_capacitance_mf_cm2: float = Field(
        ...,
        ge=0.1,
        le=2000.0,
        description="Areal capacitance in mF/cm²",
        examples=[350.5],
    )
    esr_ohm: float = Field(
        ...,
        ge=0.01,
        le=1000.0,
        description="Equivalent series resistance in Ω",
        examples=[2.5],
    )
    rate_capability_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Rate capability (capacitance retention at high rate)",
        examples=[85.0],
    )
    cycle_life_cycles: int = Field(
        ...,
        ge=100,
        le=100000,
        description="Cycle life (cycles to 80% retention)",
        examples=[10000],
    )


# ============================================================================
# Request Schemas
# ============================================================================


class PredictionRequest(DeviceCompositionBase, StructuralPropertiesBase, OpticalPropertiesBase):
    """
    Request schema for single device prediction.
    
    Example:
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
    """

    model_config = ConfigDict(use_enum_values=True)


class BatchPredictionRequest(BaseModel):
    """
    Request schema for batch predictions.
    
    Example:
        ```json
        {
            "devices": [
                {"mxene_type": "Ti3C2Tx", "thickness_um": 5.0, ...},
                {"mxene_type": "Mo2CTx", "thickness_um": 10.0, ...}
            ]
        }
        ```
    """

    devices: list[PredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of devices to predict (max 100)",
    )

    @field_validator("devices")
    @classmethod
    def validate_batch_size(cls, v: list[PredictionRequest]) -> list[PredictionRequest]:
        """Validate batch size."""
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 devices")
        return v


class DeviceCreate(
    DeviceCompositionBase,
    StructuralPropertiesBase,
    OpticalPropertiesBase,
    PerformanceMetricsBase,
):
    """
    Request schema for creating new training data.
    
    Example:
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
            "source": "DOI:10.1234/example",
            "notes": "Experimental data from lab"
        }
        ```
    """

    source: Optional[str] = Field(
        None,
        max_length=200,
        description="Data source (e.g., DOI, lab notebook)",
        examples=["DOI:10.1234/example"],
    )
    notes: Optional[str] = Field(
        None,
        max_length=1000,
        description="Additional notes",
        examples=["Experimental data from lab"],
    )

    model_config = ConfigDict(use_enum_values=True)


# ============================================================================
# Response Schemas
# ============================================================================


class UncertaintyInterval(BaseModel):
    """Uncertainty quantification for a single metric."""

    value: float = Field(..., description="Predicted value")
    lower_ci: float = Field(..., description="Lower 95% confidence interval")
    upper_ci: float = Field(..., description="Upper 95% confidence interval")
    confidence: ConfidenceLevel = Field(..., description="Confidence level")

    model_config = ConfigDict(use_enum_values=True)


class PredictionResult(BaseModel):
    """
    Prediction result with uncertainty quantification.
    
    Example:
        ```json
        {
            "areal_capacitance": {
                "value": 350.5,
                "lower_ci": 320.0,
                "upper_ci": 381.0,
                "confidence": "high"
            },
            "esr": {...},
            "rate_capability": {...},
            "cycle_life": {...},
            "overall_confidence": "high",
            "confidence_score": 0.92,
            "model_version": "v0.1.0-dummy",
            "prediction_time_ms": 15.3,
            "request_id": "req_abc123"
        }
        ```
    """

    areal_capacitance: UncertaintyInterval = Field(
        ...,
        description="Areal capacitance prediction (mF/cm²)",
    )
    esr: UncertaintyInterval = Field(
        ...,
        description="ESR prediction (Ω)",
    )
    rate_capability: UncertaintyInterval = Field(
        ...,
        description="Rate capability prediction (%)",
    )
    cycle_life: UncertaintyInterval = Field(
        ...,
        description="Cycle life prediction (cycles)",
    )
    overall_confidence: ConfidenceLevel = Field(
        ...,
        description="Overall prediction confidence",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Numerical confidence score",
    )
    model_version: str = Field(..., description="ML model version")
    prediction_time_ms: float = Field(..., description="Computation time in ms")
    request_id: Optional[str] = Field(None, description="Request tracking ID")

    model_config = ConfigDict(use_enum_values=True)


class BatchPredictionResult(BaseModel):
    """
    Batch prediction results.
    
    Example:
        ```json
        {
            "predictions": [
                {"areal_capacitance": {...}, ...},
                {"areal_capacitance": {...}, ...}
            ],
            "total_count": 2,
            "successful_count": 2,
            "failed_count": 0,
            "total_time_ms": 30.5
        }
        ```
    """

    predictions: list[PredictionResult] = Field(..., description="List of predictions")
    total_count: int = Field(..., description="Total number of requests")
    successful_count: int = Field(..., description="Number of successful predictions")
    failed_count: int = Field(..., description="Number of failed predictions")
    total_time_ms: float = Field(..., description="Total computation time")


class DeviceResponse(
    DeviceCompositionBase,
    StructuralPropertiesBase,
    OpticalPropertiesBase,
    PerformanceMetricsBase,
):
    """
    Response schema for device data.
    
    Example:
        ```json
        {
            "id": 1,
            "mxene_type": "Ti3C2Tx",
            "terminations": "O",
            "electrolyte": "H2SO4",
            "thickness_um": 5.0,
            "areal_capacitance_mf_cm2": 350.5,
            "esr_ohm": 2.5,
            "rate_capability_percent": 85.0,
            "cycle_life_cycles": 10000,
            "source": "DOI:10.1234/example",
            "created_at": "2024-01-15T10:30:00",
            "updated_at": "2024-01-15T10:30:00"
        }
        ```
    """

    id: int = Field(..., description="Device ID")
    source: Optional[str] = Field(None, description="Data source")
    notes: Optional[str] = Field(None, description="Additional notes")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class DeviceListResponse(BaseModel):
    """
    Paginated list of devices.
    
    Example:
        ```json
        {
            "devices": [{...}, {...}],
            "total": 300,
            "page": 1,
            "page_size": 50,
            "total_pages": 6
        }
        ```
    """

    devices: list[DeviceResponse] = Field(..., description="List of devices")
    total: int = Field(..., description="Total number of devices")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")


class ModelMetrics(BaseModel):
    """
    ML model performance metrics.
    
    Example:
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
    """

    model_version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Model algorithm")
    metrics: dict[str, float] = Field(..., description="Performance metrics")
    training_samples: int = Field(..., description="Number of training samples")
    test_samples: int = Field(..., description="Number of test samples")
    trained_at: datetime = Field(..., description="Training timestamp")
    is_active: bool = Field(..., description="Whether this is the active model")


class HealthCheckResponse(BaseModel):
    """
    Health check response.
    
    Example:
        ```json
        {
            "status": "healthy",
            "version": "0.1.0",
            "database": "connected",
            "model": "loaded"
        }
        ```
    """

    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    database: str = Field(..., description="Database connection status")
    model: str = Field(..., description="ML model status")


class ErrorResponse(BaseModel):
    """
    Error response schema.
    
    Example:
        ```json
        {
            "error": "Validation failed",
            "error_code": "VALIDATION_ERROR",
            "details": {"field": "thickness_um", "message": "Value out of range"},
            "request_id": "req_abc123"
        }
        ```
    """

    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: dict[str, str] = Field(default_factory=dict, description="Additional details")
    request_id: Optional[str] = Field(None, description="Request tracking ID")
