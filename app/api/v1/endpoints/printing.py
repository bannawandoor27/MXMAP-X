"""Printing/Process-Aware Design API endpoints."""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db
from app.ml.printing_surrogates import PrintingSurrogate
from app.models.database import PrintingCalibration, PrintingRun


# Pydantic schemas
class EnvironmentInput(BaseModel):
    """Environmental conditions."""
    temp_C: float = Field(25.0, ge=15, le=40)
    rh_pct: float = Field(45.0, ge=20, le=80)


class PostTreatmentInput(BaseModel):
    """Post-treatment parameters."""
    anneal_C: Optional[float] = Field(None, ge=80, le=300)
    anneal_min: Optional[float] = Field(None, ge=5, le=180)
    press_MPa: Optional[float] = Field(None, ge=0.1, le=50)


class PrintingRecommendRequest(BaseModel):
    """Request for printing recommendations."""
    process: str = Field(..., pattern="^(gravure|screen|inkjet|slot-die|doctor-blade)$")
    mxene_type: str = "Ti3C2Tx"
    flake_size_um: float = Field(2.0, ge=0.1, le=20.0)
    target_thickness_nm: float = Field(200.0, ge=50, le=2000)
    target_transmittance_550nm: float = Field(0.85, ge=0.1, le=1.0)
    substrate: str = "PET"
    environment: EnvironmentInput = EnvironmentInput()
    post_treatment: PostTreatmentInput = PostTreatmentInput()


class InkWindowOutput(BaseModel):
    """Ink formulation window."""
    solids_wt_pct: List[float]
    viscosity_mPas: List[float]
    surface_tension_mNpm: List[float]
    flake_distribution_um: Dict[str, float]
    additives: List[str]


class PrintedFilmOutput(BaseModel):
    """Printed film properties."""
    sheet_res_ohm_sq: float
    transmittance_550nm: float
    haacke_FoM: float
    rs_t_curve: Dict[str, List[float]]


class PostTreatmentEffectOutput(BaseModel):
    """Post-treatment effects."""
    delta_rs_pct: float
    delta_T_pct: float
    roughness_change_nm: float


class PrintingRecommendResponse(BaseModel):
    """Response for printing recommendations."""
    ink_window: InkWindowOutput
    printed_film: PrintedFilmOutput
    post_treatment_effect: PostTreatmentEffectOutput
    manufacturability_score: int
    assumptions: str


class PrintingEstimateRequest(BaseModel):
    """Request for film property estimation."""
    process: str = Field(..., pattern="^(gravure|screen|inkjet|slot-die|doctor-blade)$")
    solids_wt_pct: float = Field(..., ge=0.1, le=20)
    viscosity_mPas: float = Field(..., ge=1, le=10000)
    flake_size_um: float = Field(2.0, ge=0.1, le=20)
    passes: int = Field(1, ge=1, le=20)
    mxene_type: str = "Ti3C2Tx"
    post_treatment: PostTreatmentInput = PostTreatmentInput()


class PresetsOutput(BaseModel):
    """Process presets."""
    processes: Dict[str, Dict[str, Any]]
    substrates: Dict[str, Dict[str, Any]]
    typical_targets: Dict[str, Any]


router = APIRouter()
surrogate = PrintingSurrogate()


@router.post("/recommend", response_model=PrintingRecommendResponse)
async def recommend_printing(
    request: PrintingRecommendRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Recommend ink formulation and predict printed film properties.
    
    Returns ink window, Rs-T trade-off, post-treatment effects, and manufacturability score.
    """
    start_time = time.time()
    
    try:
        # Get ink window
        ink_window = surrogate.get_ink_window(
            process=request.process,
            mxene_type=request.mxene_type,
            flake_size_um=request.flake_size_um,
            target_thickness_nm=request.target_thickness_nm,
        )
        
        # Estimate printed film at mid-range solids/viscosity
        mid_solids = np.mean(ink_window.solids_wt_pct)
        mid_viscosity = np.mean(ink_window.viscosity_mPas)
        
        # Estimate passes needed
        params = surrogate.PROCESS_PARAMS[request.process]
        passes_needed = int(np.ceil(request.target_thickness_nm / params["typical_thickness_per_pass"]))
        passes_needed = max(1, min(10, passes_needed))
        
        film = surrogate.estimate_printed_film(
            process=request.process,
            solids_wt_pct=mid_solids,
            viscosity_mPas=mid_viscosity,
            flake_size_um=request.flake_size_um,
            passes=passes_needed,
            mxene_type=request.mxene_type,
        )
        
        # Generate Rs-T curve
        rs_curve, t_curve = surrogate.generate_rs_t_curve(
            process=request.process,
            solids_wt_pct=mid_solids,
            viscosity_mPas=mid_viscosity,
            flake_size_um=request.flake_size_um,
            mxene_type=request.mxene_type,
            n_points=20,
        )
        
        # Post-treatment effects
        post_effect = surrogate.estimate_post_treatment_effect(
            anneal_C=request.post_treatment.anneal_C,
            anneal_min=request.post_treatment.anneal_min,
            press_MPa=request.post_treatment.press_MPa,
            initial_Rs=film.sheet_res_ohm_sq,
            initial_T=film.transmittance_550nm,
        )
        
        # Apply post-treatment
        final_rs = film.sheet_res_ohm_sq * (1 + post_effect.delta_rs_pct / 100)
        final_t = film.transmittance_550nm * (1 + post_effect.delta_T_pct / 100)
        final_fom = (final_t ** 10) / final_rs if final_rs > 0 else 0
        
        # Manufacturability score
        manuf_score = surrogate.calculate_manufacturability_score(
            process=request.process,
            solids_wt_pct=mid_solids,
            viscosity_mPas=mid_viscosity,
            flake_d90_um=ink_window.flake_distribution_um["d90"],
            passes=passes_needed,
            anneal_C=request.post_treatment.anneal_C,
            press_MPa=request.post_treatment.press_MPa,
            substrate=request.substrate,
        )
        
        # Build response
        response = PrintingRecommendResponse(
            ink_window=InkWindowOutput(
                solids_wt_pct=list(ink_window.solids_wt_pct),
                viscosity_mPas=list(ink_window.viscosity_mPas),
                surface_tension_mNpm=list(ink_window.surface_tension_mNpm),
                flake_distribution_um=ink_window.flake_distribution_um,
                additives=ink_window.additives,
            ),
            printed_film=PrintedFilmOutput(
                sheet_res_ohm_sq=final_rs,
                transmittance_550nm=final_t,
                haacke_FoM=final_fom,
                rs_t_curve={
                    "rs_ohm_sq": rs_curve.tolist(),
                    "T_550nm": t_curve.tolist(),
                },
            ),
            post_treatment_effect=PostTreatmentEffectOutput(
                delta_rs_pct=post_effect.delta_rs_pct,
                delta_T_pct=post_effect.delta_T_pct,
                roughness_change_nm=post_effect.roughness_change_nm,
            ),
            manufacturability_score=manuf_score,
            assumptions=(
                f"Process: {request.process}, {passes_needed} passes. "
                f"Mid-range ink: {mid_solids:.1f} wt%, {mid_viscosity:.0f} mPa·s. "
                f"Post-treatment improves contact. "
                f"Computation time: {(time.time() - start_time) * 1000:.1f} ms."
            ),
        )
        
        # Store run in database
        import json
        run = PrintingRun(
            input_json=json.dumps(request.dict()),
            output_json=json.dumps(response.dict()),
            manufacturability_score=manuf_score,
        )
        db.add(run)
        await db.commit()
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@router.post("/estimate", response_model=PrintingRecommendResponse)
async def estimate_printing(
    request: PrintingEstimateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Estimate printed film properties for explicit ink formulation.
    
    Returns Rs-T trade-off and manufacturability score.
    """
    start_time = time.time()
    
    try:
        # Estimate film
        film = surrogate.estimate_printed_film(
            process=request.process,
            solids_wt_pct=request.solids_wt_pct,
            viscosity_mPas=request.viscosity_mPas,
            flake_size_um=request.flake_size_um,
            passes=request.passes,
            mxene_type=request.mxene_type,
        )
        
        # Generate Rs-T curve
        rs_curve, t_curve = surrogate.generate_rs_t_curve(
            process=request.process,
            solids_wt_pct=request.solids_wt_pct,
            viscosity_mPas=request.viscosity_mPas,
            flake_size_um=request.flake_size_um,
            mxene_type=request.mxene_type,
            n_points=20,
        )
        
        # Post-treatment effects
        post_effect = surrogate.estimate_post_treatment_effect(
            anneal_C=request.post_treatment.anneal_C,
            anneal_min=request.post_treatment.anneal_min,
            press_MPa=request.post_treatment.press_MPa,
            initial_Rs=film.sheet_res_ohm_sq,
            initial_T=film.transmittance_550nm,
        )
        
        # Apply post-treatment
        final_rs = film.sheet_res_ohm_sq * (1 + post_effect.delta_rs_pct / 100)
        final_t = film.transmittance_550nm * (1 + post_effect.delta_T_pct / 100)
        final_fom = (final_t ** 10) / final_rs if final_rs > 0 else 0
        
        # Get ink window for reference
        ink_window = surrogate.get_ink_window(
            process=request.process,
            mxene_type=request.mxene_type,
            flake_size_um=request.flake_size_um,
            target_thickness_nm=film.thickness_nm,
        )
        
        # Manufacturability score
        manuf_score = surrogate.calculate_manufacturability_score(
            process=request.process,
            solids_wt_pct=request.solids_wt_pct,
            viscosity_mPas=request.viscosity_mPas,
            flake_d90_um=request.flake_size_um * 3.0,
            passes=request.passes,
            anneal_C=request.post_treatment.anneal_C,
            press_MPa=request.post_treatment.press_MPa,
            substrate="PET",
        )
        
        response = PrintingRecommendResponse(
            ink_window=InkWindowOutput(
                solids_wt_pct=list(ink_window.solids_wt_pct),
                viscosity_mPas=list(ink_window.viscosity_mPas),
                surface_tension_mNpm=list(ink_window.surface_tension_mNpm),
                flake_distribution_um=ink_window.flake_distribution_um,
                additives=ink_window.additives,
            ),
            printed_film=PrintedFilmOutput(
                sheet_res_ohm_sq=final_rs,
                transmittance_550nm=final_t,
                haacke_FoM=final_fom,
                rs_t_curve={
                    "rs_ohm_sq": rs_curve.tolist(),
                    "T_550nm": t_curve.tolist(),
                },
            ),
            post_treatment_effect=PostTreatmentEffectOutput(
                delta_rs_pct=post_effect.delta_rs_pct,
                delta_T_pct=post_effect.delta_T_pct,
                roughness_change_nm=post_effect.roughness_change_nm,
            ),
            manufacturability_score=manuf_score,
            assumptions=(
                f"Explicit formulation: {request.solids_wt_pct:.1f} wt%, "
                f"{request.viscosity_mPas:.0f} mPa·s, {request.passes} passes. "
                f"Coverage: {film.coverage_factor:.2f}. "
                f"Computation time: {(time.time() - start_time) * 1000:.1f} ms."
            ),
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Estimation failed: {str(e)}")


@router.post("/fit")
async def fit_printing_model(
    file: UploadFile = File(...),
    process: str = "screen",
    db: AsyncSession = Depends(get_db),
):
    """
    Fit printing surrogate from experimental data CSV.
    
    CSV columns: process, solids_wt_pct, viscosity_mPas, flake_d50_um, 
                 passes, post_anneal_C, post_press_MPa, thickness_nm, 
                 Rs_ohm_sq, T_550nm
    """
    try:
        # Read CSV
        contents = await file.read()
        import io
        df = pd.read_csv(io.BytesIO(contents))
        
        # Validate columns
        required_cols = [
            "solids_wt_pct", "viscosity_mPas", "flake_d50_um",
            "Rs_ohm_sq", "T_550nm"
        ]
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain columns: {required_cols}"
            )
        
        # Convert to list of dicts
        data = df.to_dict('records')
        
        # Fit model
        metrics = surrogate.fit_from_data(data, process)
        
        # Store calibration in database
        import json
        calibration = PrintingCalibration(
            process=process,
            version="v1.0",
            coeffs_json=json.dumps({"rs_scale": 1.0, "t_scale": 1.0}),
            metrics_json=json.dumps(metrics),
        )
        db.add(calibration)
        await db.commit()
        
        return {
            "status": "success",
            "process": process,
            "metrics": metrics,
            "n_samples": len(data),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fitting failed: {str(e)}")


@router.get("/presets", response_model=PresetsOutput)
async def get_printing_presets():
    """Get printing process presets and typical targets."""
    return PresetsOutput(
        processes={
            "gravure": {
                "solids_range": [8, 15],
                "viscosity_range": [50, 200],
                "typical_thickness_per_pass": 150,
                "web_speed_m_min": [10, 100],
                "description": "High-speed roll-to-roll, fine features"
            },
            "screen": {
                "solids_range": [5, 12],
                "viscosity_range": [800, 3000],
                "typical_thickness_per_pass": 300,
                "mesh_count": [200, 400],
                "description": "Thick films, simple patterns"
            },
            "inkjet": {
                "solids_range": [0.5, 3],
                "viscosity_range": [5, 20],
                "typical_thickness_per_pass": 50,
                "nozzle_diameter_um": [20, 80],
                "description": "Digital, fine features, thin films"
            },
            "slot-die": {
                "solids_range": [3, 10],
                "viscosity_range": [100, 500],
                "typical_thickness_per_pass": 200,
                "web_speed_m_min": [5, 50],
                "description": "Uniform coating, roll-to-roll"
            },
            "doctor-blade": {
                "solids_range": [5, 15],
                "viscosity_range": [200, 1000],
                "typical_thickness_per_pass": 250,
                "blade_gap_um": [50, 500],
                "description": "Lab-scale, flexible thickness"
            },
        },
        substrates={
            "PET": {"Tg_C": 120, "surface_energy_mN_m": 45, "typical_thickness_um": 125},
            "PI": {"Tg_C": 300, "surface_energy_mN_m": 40, "typical_thickness_um": 50},
            "glass": {"Tg_C": 500, "surface_energy_mN_m": 70, "typical_thickness_um": 700},
            "paper": {"Tg_C": 150, "surface_energy_mN_m": 35, "typical_thickness_um": 100},
        },
        typical_targets={
            "transparent_conductor": {
                "Rs_ohm_sq": [10, 100],
                "T_550nm": [0.8, 0.95],
                "haacke_FoM_min": 0.001,
            },
            "opaque_electrode": {
                "Rs_ohm_sq": [1, 10],
                "T_550nm": [0.0, 0.3],
            },
            "EMI_shielding": {
                "Rs_ohm_sq": [0.1, 5],
                "thickness_nm": [500, 2000],
            },
        },
    )
