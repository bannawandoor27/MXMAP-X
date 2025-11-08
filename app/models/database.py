"""SQLAlchemy database models for MXMAP-X."""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    ForeignKey,
    Index,
    CheckConstraint,
)
from sqlalchemy.orm import relationship
from app.db.session import Base


class Device(Base):
    """
    Training data for MXene supercapacitor devices.
    
    Stores material composition, processing parameters, and measured performance.
    """

    __tablename__ = "devices"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True)

    # Material Composition
    mxene_type = Column(
        String(50),
        nullable=False,
        comment="MXene formula (e.g., Ti3C2Tx, Mo2CTx)",
    )
    terminations = Column(
        String(50),
        nullable=False,
        comment="Surface terminations (e.g., O, OH, F, mixed)",
    )
    electrolyte = Column(
        String(100),
        nullable=False,
        comment="Electrolyte type (e.g., H2SO4, KOH, ionic_liquid)",
    )
    electrolyte_concentration = Column(
        Float,
        nullable=True,
        comment="Electrolyte concentration in M",
    )

    # Processing Parameters
    thickness_um = Column(
        Float,
        nullable=False,
        comment="Film thickness in micrometers",
    )
    deposition_method = Column(
        String(50),
        nullable=False,
        comment="Deposition method (e.g., vacuum_filtration, spray_coating)",
    )
    annealing_temp_c = Column(
        Float,
        nullable=True,
        comment="Annealing temperature in Celsius",
    )
    annealing_time_min = Column(
        Float,
        nullable=True,
        comment="Annealing time in minutes",
    )

    # Structural Properties
    interlayer_spacing_nm = Column(
        Float,
        nullable=True,
        comment="Interlayer spacing in nanometers",
    )
    specific_surface_area_m2g = Column(
        Float,
        nullable=True,
        comment="Specific surface area in m²/g",
    )
    pore_volume_cm3g = Column(
        Float,
        nullable=True,
        comment="Pore volume in cm³/g",
    )

    # Optical Properties (may be missing)
    optical_transmittance = Column(
        Float,
        nullable=True,
        comment="Optical transmittance percentage",
    )
    sheet_resistance_ohm_sq = Column(
        Float,
        nullable=True,
        comment="Sheet resistance in Ω/sq",
    )

    # Performance Metrics
    areal_capacitance_mf_cm2 = Column(
        Float,
        nullable=False,
        comment="Areal capacitance in mF/cm²",
    )
    esr_ohm = Column(
        Float,
        nullable=False,
        comment="Equivalent series resistance in Ω",
    )
    rate_capability_percent = Column(
        Float,
        nullable=False,
        comment="Rate capability (capacitance retention at high rate)",
    )
    cycle_life_cycles = Column(
        Integer,
        nullable=False,
        comment="Cycle life (cycles to 80% retention)",
    )

    # Metadata
    source = Column(
        String(200),
        nullable=True,
        comment="Data source (e.g., DOI, lab notebook)",
    )
    notes = Column(
        Text,
        nullable=True,
        comment="Additional notes",
    )
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    predictions = relationship("Prediction", back_populates="device", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        CheckConstraint("thickness_um > 0", name="check_thickness_positive"),
        CheckConstraint("areal_capacitance_mf_cm2 > 0", name="check_capacitance_positive"),
        CheckConstraint("esr_ohm > 0", name="check_esr_positive"),
        CheckConstraint(
            "rate_capability_percent >= 0 AND rate_capability_percent <= 100",
            name="check_rate_capability_range",
        ),
        CheckConstraint("cycle_life_cycles > 0", name="check_cycle_life_positive"),
        Index("idx_mxene_type", "mxene_type"),
        Index("idx_electrolyte", "electrolyte"),
        Index("idx_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<Device(id={self.id}, mxene={self.mxene_type}, "
            f"capacitance={self.areal_capacitance_mf_cm2:.2f} mF/cm²)>"
        )


class Prediction(Base):
    """
    ML model predictions with uncertainty quantification.
    
    Stores prediction results, confidence intervals, and model metadata.
    """

    __tablename__ = "predictions"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign Key
    device_id = Column(
        Integer,
        ForeignKey("devices.id", ondelete="CASCADE"),
        nullable=True,
        comment="Reference to training device (if applicable)",
    )

    # Input Parameters (denormalized for query performance)
    mxene_type = Column(String(50), nullable=False)
    terminations = Column(String(50), nullable=False)
    electrolyte = Column(String(100), nullable=False)
    thickness_um = Column(Float, nullable=False)

    # Predictions with Uncertainty
    predicted_capacitance = Column(
        Float,
        nullable=False,
        comment="Predicted areal capacitance in mF/cm²",
    )
    capacitance_lower_ci = Column(
        Float,
        nullable=False,
        comment="Lower 95% confidence interval",
    )
    capacitance_upper_ci = Column(
        Float,
        nullable=False,
        comment="Upper 95% confidence interval",
    )

    predicted_esr = Column(
        Float,
        nullable=False,
        comment="Predicted ESR in Ω",
    )
    esr_lower_ci = Column(Float, nullable=False)
    esr_upper_ci = Column(Float, nullable=False)

    predicted_rate_capability = Column(
        Float,
        nullable=False,
        comment="Predicted rate capability in %",
    )
    rate_capability_lower_ci = Column(Float, nullable=False)
    rate_capability_upper_ci = Column(Float, nullable=False)

    predicted_cycle_life = Column(
        Integer,
        nullable=False,
        comment="Predicted cycle life in cycles",
    )
    cycle_life_lower_ci = Column(Integer, nullable=False)
    cycle_life_upper_ci = Column(Integer, nullable=False)

    # Confidence Metrics
    overall_confidence = Column(
        String(20),
        nullable=False,
        comment="Overall confidence level (high/medium/low)",
    )
    confidence_score = Column(
        Float,
        nullable=False,
        comment="Numerical confidence score (0-1)",
    )

    # Model Metadata
    model_version = Column(
        String(50),
        nullable=False,
        comment="ML model version used",
    )
    prediction_time_ms = Column(
        Float,
        nullable=True,
        comment="Prediction computation time in milliseconds",
    )

    # Request Tracking
    request_id = Column(
        String(100),
        nullable=True,
        index=True,
        comment="Request ID for debugging",
    )

    # Timestamps
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    device = relationship("Device", back_populates="predictions")

    # Indexes
    __table_args__ = (
        Index("idx_prediction_created_at", "created_at"),
        Index("idx_prediction_confidence", "overall_confidence"),
        Index("idx_prediction_model_version", "model_version"),
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction(id={self.id}, capacitance={self.predicted_capacitance:.2f}, "
            f"confidence={self.overall_confidence})>"
        )


class TrainingMetadata(Base):
    """
    Metadata about ML model training runs.
    
    Tracks model performance, hyperparameters, and training history.
    """

    __tablename__ = "training_metadata"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True)

    # Model Information
    model_version = Column(
        String(50),
        nullable=False,
        unique=True,
        comment="Unique model version identifier",
    )
    model_type = Column(
        String(50),
        nullable=False,
        comment="Model algorithm (e.g., xgboost, random_forest)",
    )

    # Training Metrics
    train_r2_capacitance = Column(Float, nullable=True)
    train_r2_esr = Column(Float, nullable=True)
    train_r2_rate_capability = Column(Float, nullable=True)
    train_r2_cycle_life = Column(Float, nullable=True)

    test_r2_capacitance = Column(Float, nullable=True)
    test_r2_esr = Column(Float, nullable=True)
    test_r2_rate_capability = Column(Float, nullable=True)
    test_r2_cycle_life = Column(Float, nullable=True)

    train_rmse_capacitance = Column(Float, nullable=True)
    test_rmse_capacitance = Column(Float, nullable=True)

    # Training Configuration
    training_samples = Column(Integer, nullable=False)
    test_samples = Column(Integer, nullable=False)
    hyperparameters = Column(Text, nullable=True, comment="JSON-encoded hyperparameters")

    # Timestamps
    trained_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(
        Integer,
        default=0,
        nullable=False,
        comment="1 if this is the active model, 0 otherwise",
    )

    # Indexes
    __table_args__ = (
        Index("idx_model_version", "model_version"),
        Index("idx_is_active", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<TrainingMetadata(version={self.model_version}, active={bool(self.is_active)})>"
