"""Dummy predictor for testing and development."""

import time
import uuid
from typing import Any
import numpy as np
from app.ml.predictor import BasePredictor
from app.models.schemas import (
    PredictionRequest,
    PredictionResult,
    UncertaintyInterval,
    ConfidenceLevel,
)
from app.core.exceptions import PredictionError


class DummyPredictor(BasePredictor):
    """
    Placeholder ML model that returns physics-informed predictions.
    
    This dummy model uses simple heuristics based on material properties
    to generate realistic predictions with uncertainty quantification.
    It's designed for testing the API before real ML models are trained.
    """

    def __init__(self, model_version: str = "v0.1.0-dummy", config: dict[str, Any] | None = None) -> None:
        """Initialize dummy predictor."""
        super().__init__(model_version, config or {})
        self.rng = np.random.default_rng(42)  # Reproducible randomness

    async def predict(self, request: PredictionRequest) -> PredictionResult:
        """
        Generate physics-informed dummy prediction.
        
        Heuristics:
        - Thicker films → higher capacitance, worse rate capability
        - Ti3C2Tx with O terminations → baseline performance
        - Ionic liquids → better high-temp performance
        - H2SO4 → higher capacitance but lower cycle life
        """
        start_time = time.time()

        try:
            # Generate predictions based on physics-informed heuristics
            capacitance = self._predict_capacitance(request)
            esr = self._predict_esr(request)
            rate_capability = self._predict_rate_capability(request)
            cycle_life = self._predict_cycle_life(request)

            # Calculate overall confidence
            confidence_score = self._calculate_confidence(request)
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

    def _predict_capacitance(self, request: PredictionRequest) -> UncertaintyInterval:
        """
        Predict areal capacitance (mF/cm²).
        
        Heuristics:
        - Base: 200-400 mF/cm² for Ti3C2Tx
        - Thickness effect: +20 mF/cm² per μm
        - Electrolyte: H2SO4 +50, KOH +30, ionic_liquid +20
        - Surface area: +0.5 mF/cm² per m²/g
        """
        # Base capacitance by MXene type
        base_capacitance = {
            "Ti3C2Tx": 300.0,
            "Mo2CTx": 250.0,
            "V2CTx": 280.0,
            "Ti2CTx": 220.0,
            "Nb2CTx": 200.0,
            "Ta4C3Tx": 240.0,
            "Ti3CNTx": 310.0,
        }

        capacitance = base_capacitance.get(request.mxene_type, 250.0)

        # Thickness effect (thicker → higher capacitance)
        capacitance += request.thickness_um * 15.0

        # Electrolyte effect
        electrolyte_bonus = {
            "H2SO4": 50.0,
            "KOH": 30.0,
            "NaOH": 25.0,
            "ionic_liquid": 20.0,
            "EMIMBF4": 20.0,
            "PVA_H2SO4": 40.0,
            "PVA_KOH": 25.0,
            "organic": 15.0,
        }
        capacitance += electrolyte_bonus.get(request.electrolyte, 20.0)

        # Surface area effect
        if request.specific_surface_area_m2g:
            capacitance += request.specific_surface_area_m2g * 0.5

        # Annealing effect (moderate annealing improves performance)
        if request.annealing_temp_c:
            if 100 <= request.annealing_temp_c <= 200:
                capacitance *= 1.1
            elif request.annealing_temp_c > 300:
                capacitance *= 0.9

        # Add realistic noise
        noise = self.rng.normal(0, capacitance * 0.05)
        value = max(50.0, capacitance + noise)

        # Calculate uncertainty (±10-15%)
        uncertainty = value * 0.12
        lower_ci = max(10.0, value - uncertainty)
        upper_ci = value + uncertainty

        confidence = self._get_confidence_level(0.85)

        return UncertaintyInterval(
            value=round(value, 2),
            lower_ci=round(lower_ci, 2),
            upper_ci=round(upper_ci, 2),
            confidence=confidence,
        )

    def _predict_esr(self, request: PredictionRequest) -> UncertaintyInterval:
        """
        Predict equivalent series resistance (Ω).
        
        Heuristics:
        - Base: 1-5 Ω
        - Thickness effect: +0.1 Ω per μm
        - Ionic liquids: higher ESR
        - Sheet resistance correlation
        """
        # Base ESR
        esr = 2.0

        # Thickness effect (thicker → higher ESR)
        esr += request.thickness_um * 0.08

        # Electrolyte effect
        if request.electrolyte in ["ionic_liquid", "EMIMBF4"]:
            esr *= 1.5
        elif request.electrolyte in ["H2SO4", "KOH"]:
            esr *= 0.8

        # Sheet resistance correlation
        if request.sheet_resistance_ohm_sq:
            esr += request.sheet_resistance_ohm_sq * 0.01

        # Add noise
        noise = self.rng.normal(0, esr * 0.08)
        value = max(0.1, esr + noise)

        # Uncertainty
        uncertainty = value * 0.15
        lower_ci = max(0.05, value - uncertainty)
        upper_ci = value + uncertainty

        confidence = self._get_confidence_level(0.82)

        return UncertaintyInterval(
            value=round(value, 3),
            lower_ci=round(lower_ci, 3),
            upper_ci=round(upper_ci, 3),
            confidence=confidence,
        )

    def _predict_rate_capability(self, request: PredictionRequest) -> UncertaintyInterval:
        """
        Predict rate capability (%).
        
        Heuristics:
        - Base: 70-90%
        - Thickness effect: -1% per μm (thicker → worse)
        - Interlayer spacing: larger spacing → better
        - Pore volume: higher → better
        """
        # Base rate capability
        rate_cap = 85.0

        # Thickness effect (thicker → worse rate capability)
        rate_cap -= request.thickness_um * 0.8

        # Interlayer spacing effect
        if request.interlayer_spacing_nm:
            if request.interlayer_spacing_nm > 1.5:
                rate_cap += 5.0
            elif request.interlayer_spacing_nm < 1.0:
                rate_cap -= 3.0

        # Pore volume effect
        if request.pore_volume_cm3g:
            rate_cap += request.pore_volume_cm3g * 10.0

        # Deposition method effect
        if request.deposition_method == "spray_coating":
            rate_cap += 3.0
        elif request.deposition_method == "drop_casting":
            rate_cap -= 2.0

        # Add noise
        noise = self.rng.normal(0, 3.0)
        value = np.clip(rate_cap + noise, 30.0, 98.0)

        # Uncertainty
        uncertainty = 5.0
        lower_ci = max(20.0, value - uncertainty)
        upper_ci = min(100.0, value + uncertainty)

        confidence = self._get_confidence_level(0.78)

        return UncertaintyInterval(
            value=round(value, 2),
            lower_ci=round(lower_ci, 2),
            upper_ci=round(upper_ci, 2),
            confidence=confidence,
        )

    def _predict_cycle_life(self, request: PredictionRequest) -> UncertaintyInterval:
        """
        Predict cycle life (cycles to 80% retention).
        
        Heuristics:
        - Base: 5000-15000 cycles
        - H2SO4: lower cycle life (acidic)
        - Ionic liquids: higher cycle life
        - Annealing: improves stability
        """
        # Base cycle life
        cycle_life = 10000

        # Electrolyte effect
        if request.electrolyte == "H2SO4":
            cycle_life *= 0.7
        elif request.electrolyte in ["ionic_liquid", "EMIMBF4"]:
            cycle_life *= 1.3
        elif request.electrolyte in ["PVA_H2SO4", "PVA_KOH"]:
            cycle_life *= 1.5  # Solid-state better stability

        # Annealing effect
        if request.annealing_temp_c and request.annealing_temp_c > 100:
            cycle_life *= 1.2

        # MXene type effect
        if request.mxene_type == "Ti3C2Tx":
            cycle_life *= 1.1  # Most stable
        elif request.mxene_type == "V2CTx":
            cycle_life *= 0.9  # Less stable

        # Add noise
        noise = self.rng.normal(0, cycle_life * 0.1)
        value = int(max(1000, cycle_life + noise))

        # Uncertainty
        uncertainty = int(value * 0.15)
        lower_ci = max(500, value - uncertainty)
        upper_ci = min(50000, value + uncertainty)

        confidence = self._get_confidence_level(0.75)

        return UncertaintyInterval(
            value=value,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            confidence=confidence,
        )

    def _calculate_confidence(self, request: PredictionRequest) -> float:
        """
        Calculate overall confidence score based on data completeness.
        
        More complete data → higher confidence
        """
        confidence = 0.7  # Base confidence

        # Bonus for optional parameters
        if request.electrolyte_concentration is not None:
            confidence += 0.05
        if request.annealing_temp_c is not None:
            confidence += 0.05
        if request.interlayer_spacing_nm is not None:
            confidence += 0.05
        if request.specific_surface_area_m2g is not None:
            confidence += 0.05
        if request.pore_volume_cm3g is not None:
            confidence += 0.03
        if request.optical_transmittance is not None:
            confidence += 0.02
        if request.sheet_resistance_ohm_sq is not None:
            confidence += 0.02

        # Penalty for extreme values (extrapolation)
        if request.thickness_um > 30.0 or request.thickness_um < 1.0:
            confidence -= 0.1
        if request.annealing_temp_c and request.annealing_temp_c > 300:
            confidence -= 0.05

        return np.clip(confidence, 0.0, 1.0)

    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert numerical confidence to categorical level."""
        if score >= 0.9:
            return ConfidenceLevel.HIGH
        elif score >= 0.7:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def get_feature_importance(self) -> dict[str, float]:
        """Get dummy feature importance scores."""
        return {
            "thickness_um": 0.25,
            "mxene_type": 0.20,
            "electrolyte": 0.18,
            "specific_surface_area_m2g": 0.12,
            "interlayer_spacing_nm": 0.10,
            "annealing_temp_c": 0.08,
            "deposition_method": 0.07,
        }

    def get_model_info(self) -> dict[str, Any]:
        """Get model metadata."""
        return {
            "model_version": self.model_version,
            "model_type": "dummy",
            "description": "Physics-informed dummy predictor for testing",
            "features": [
                "mxene_type",
                "terminations",
                "electrolyte",
                "thickness_um",
                "deposition_method",
                "annealing_temp_c",
                "interlayer_spacing_nm",
                "specific_surface_area_m2g",
            ],
            "targets": [
                "areal_capacitance_mf_cm2",
                "esr_ohm",
                "rate_capability_percent",
                "cycle_life_cycles",
            ],
        }


# Global predictor instance
_predictor: DummyPredictor | None = None


def get_predictor() -> DummyPredictor:
    """Get or create global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = DummyPredictor()
    return _predictor
