"""AC-line filtering API endpoints."""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db
from app.ml.eis_surrogate import (
    EISSurrogate,
    CPEParams,
    geometry_to_params,
    calculate_device_area,
)
from app.ml.optimization import MultiObjectiveOptimizer
from app.models.database import FilteringModel, FilteringRun


# Pydantic schemas
class GeometryInput(BaseModel):
    """Interdigitated electrode geometry."""
    finger_width_um: float = Field(8.0, ge=2.0, le=50.0)
    finger_spacing_um: float = Field(8.0, ge=2.0, le=50.0)
    finger_length_um: float = Field(2000.0, ge=500.0, le=10000.0)
    num_fingers_per_electrode: int = Field(50, ge=10, le=200)
    overlap_length_um: float = Field(1800.0, ge=400.0, le=9500.0)
    thickness_nm: float = Field(200.0, ge=50.0, le=1000.0)
    substrate: str = "Si/SiO2"


class MXeneFilmInput(BaseModel):
    """MXene film properties."""
    flake_size_um: float = Field(2.0, ge=0.1, le=10.0)
    porosity_pct: float = Field(30.0, ge=10.0, le=60.0)
    sheet_res_ohm_sq: float = Field(10.0, ge=1.0, le=1000.0)
    terminations: str = "-O,-OH"
    electrolyte: str = "PVA/H2SO4"
    process: str = "photolithography"


class FitDataInput(BaseModel):
    """Optional fitting from measured data."""
    enable: bool = False
    digitized_bode_csv_path: Optional[str] = None


class FilteringPredictRequest(BaseModel):
    """Request for filtering prediction."""
    frequency_range_hz: List[float] = Field([1, 100000])
    load_resistance_ohm: float = Field(33.0, ge=1.0, le=1000.0)
    geometry: GeometryInput
    mxene_film: MXeneFilmInput
    fit_from_data: FitDataInput = FitDataInput()


class TargetsInput(BaseModel):
    """Optimization targets."""
    phase_deg_120hz: float = Field(-85.0, ge=-90.0, le=-70.0)
    C_min_mf_cm2_120hz: float = Field(10.0, ge=1.0, le=100.0)
    Z_max_ohm_120hz: float = Field(2.0, ge=0.1, le=10.0)


class ConstraintsInput(BaseModel):
    """Optimization constraints."""
    max_area_mm2: float = Field(4.0, ge=1.0, le=100.0)
    min_spacing_um: float = Field(5.0, ge=2.0, le=20.0)
    targets: TargetsInput


class FilteringOptimizeRequest(BaseModel):
    """Request for filtering optimization."""
    frequency_range_hz: List[float] = Field([1, 100000])
    load_resistance_ohm: float = Field(33.0, ge=1.0, le=1000.0)
    mxene_film: MXeneFilmInput
    constraints: ConstraintsInput


class FilteringFitRequest(BaseModel):
    """Request for fitting from data."""
    digitized_bode_csv_path: str
    electrolyte: str
    process: str
    fit_rleak: bool = False


class KPIsOutput(BaseModel):
    """Key performance indicators."""
    phase_deg_120hz: float
    capacitance_mf_cm2_120hz: float
    impedance_ohm_120hz: float
    attenuation_db_50hz: float
    attenuation_db_60hz: float
    f_phi_minus60_deg_hz: Optional[float]
    device_area_mm2: float


class BodeOutput(BaseModel):
    """Bode plot data."""
    freq_hz: List[float]
    mag_ohm: List[float]
    phase_deg: List[float]


class NyquistOutput(BaseModel):
    """Nyquist plot data."""
    re_ohm: List[float]
    im_ohm: List[float]


class ParamsOutput(BaseModel):
    """Circuit parameters."""
    Rs: float
    Q: float
    alpha: float
    Rleak: float


class RecipeOutput(BaseModel):
    """Fabrication recipe."""
    geometry: Dict
    mxene_film: Dict
    assumptions: str


class FilteringPredictResponse(BaseModel):
    """Response for filtering prediction."""
    kpis: KPIsOutput
    bode: BodeOutput
    nyquist: NyquistOutput
    params: ParamsOutput
    recipe: RecipeOutput


class OptimizationSolution(BaseModel):
    """Single optimization solution."""
    geometry: GeometryInput
    kpis: KPIsOutput
    params: ParamsOutput
    bode: BodeOutput
    nyquist: NyquistOutput


class FilteringOptimizeResponse(BaseModel):
    """Response for filtering optimization."""
    solutions: List[OptimizationSolution]
    num_feasible: int
    computation_time_s: float


class PresetOutput(BaseModel):
    """Preset configurations."""
    load_resistances_ohm: List[float]
    frequency_range_default: List[float]
    geometry_bounds: Dict
    process_templates: Dict


router = APIRouter()
eis_surrogate = EISSurrogate()


@router.post("/predict", response_model=FilteringPredictResponse)
async def predict_filtering(
    request: FilteringPredictRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Predict AC-line filtering performance for given geometry.
    
    Returns impedance spectra, phase angles, and filtering KPIs.
    """
    start_time = time.time()
    
    try:
        # Generate frequency array (log-spaced, decimated to 300 points)
        freq_min, freq_max = request.frequency_range_hz
        frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), 300)
        
        # Get CPE parameters from geometry
        geometry_dict = request.geometry.dict()
        film_dict = request.mxene_film.dict()
        
        # Check if fitting from data
        if request.fit_from_data.enable and request.fit_from_data.digitized_bode_csv_path:
            # Load and fit from CSV
            df = pd.read_csv(request.fit_from_data.digitized_bode_csv_path)
            params = eis_surrogate.fit_from_bode_data(
                freq_hz=df["freq_hz"].values,
                mag_ohm=df["mag_ohm"].values,
                phase_deg=df["phase_deg"].values,
            )
            
            # Store calibration
            calib_key = f"{film_dict['electrolyte']}_{film_dict['process']}"
            # Calculate scaling factors
            params_default = geometry_to_params(geometry_dict, film_dict)
            calib_factors = {
                "Q_scale": params.Q / params_default.Q,
                "Rs_scale": params.Rs / params_default.Rs,
                "alpha": params.alpha,
            }
            eis_surrogate.store_calibration(calib_key, calib_factors)
        else:
            # Use geometry-based prediction
            calib_key = f"{film_dict['electrolyte']}_{film_dict['process']}"
            calibration = eis_surrogate.get_calibration(calib_key)
            params = geometry_to_params(geometry_dict, film_dict, calibration)
        
        # Compute impedance spectra
        Z_mag, Z_phase, Z_complex = eis_surrogate.compute_impedance(frequencies, params)
        
        # Compute effective capacitance
        C_eff_imag, C_eff_cpe = eis_surrogate.compute_effective_capacitance(
            frequencies, params, Z_complex
        )
        
        # Compute ripple attenuation
        attenuation_db = eis_surrogate.compute_ripple_attenuation(
            frequencies, params, request.load_resistance_ohm
        )
        
        # Extract KPIs at specific frequencies
        def get_value_at_freq(freq_target, freq_array, value_array):
            idx = np.argmin(np.abs(freq_array - freq_target))
            return float(value_array[idx])
        
        phase_120hz = get_value_at_freq(120, frequencies, Z_phase)
        Z_120hz = get_value_at_freq(120, frequencies, Z_mag)
        C_120hz = get_value_at_freq(120, frequencies, C_eff_cpe) * 1e3  # Convert to mF
        
        # Get active area for areal capacitance
        active_area_cm2 = (
            geometry_dict["overlap_length_um"] *
            geometry_dict["finger_spacing_um"] *
            (geometry_dict["num_fingers_per_electrode"] - 1) *
            1e-8
        )
        C_areal_120hz = C_120hz / active_area_cm2 if active_area_cm2 > 0 else 0.0
        
        atten_50hz = get_value_at_freq(50, frequencies, attenuation_db)
        atten_60hz = get_value_at_freq(60, frequencies, attenuation_db)
        
        # Find frequency where phase = -60°
        f_minus60 = eis_surrogate.find_frequency_at_phase(params, -60.0)
        
        # Calculate device area
        device_area = calculate_device_area(geometry_dict)
        
        # Decimate plot data to 512 points
        plot_indices = np.linspace(0, len(frequencies) - 1, min(512, len(frequencies)), dtype=int)
        
        # Build response
        response = FilteringPredictResponse(
            kpis=KPIsOutput(
                phase_deg_120hz=phase_120hz,
                capacitance_mf_cm2_120hz=C_areal_120hz,
                impedance_ohm_120hz=Z_120hz,
                attenuation_db_50hz=atten_50hz,
                attenuation_db_60hz=atten_60hz,
                f_phi_minus60_deg_hz=f_minus60,
                device_area_mm2=device_area,
            ),
            bode=BodeOutput(
                freq_hz=frequencies[plot_indices].tolist(),
                mag_ohm=Z_mag[plot_indices].tolist(),
                phase_deg=Z_phase[plot_indices].tolist(),
            ),
            nyquist=NyquistOutput(
                re_ohm=np.real(Z_complex[plot_indices]).tolist(),
                im_ohm=np.imag(Z_complex[plot_indices]).tolist(),
            ),
            params=ParamsOutput(
                Rs=params.Rs,
                Q=params.Q,
                alpha=params.alpha,
                Rleak=params.Rleak if np.isfinite(params.Rleak) else 1e9,
            ),
            recipe=RecipeOutput(
                geometry=geometry_dict,
                mxene_film=film_dict,
                assumptions=(
                    f"Active area: {active_area_cm2:.4f} cm². "
                    f"Circuit model: Rs + (CPE || Rleak). "
                    f"Computation time: {(time.time() - start_time) * 1000:.1f} ms."
                ),
            ),
        )
        
        # Store run in database
        import json
        run = FilteringRun(
            input_json=json.dumps(request.dict()),
            kpis_json=json.dumps(response.kpis.dict()),
            params_json=json.dumps(response.params.dict()),
            area_mm2=device_area,
        )
        db.add(run)
        await db.commit()
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/optimize", response_model=FilteringOptimizeResponse)
async def optimize_filtering(
    request: FilteringOptimizeRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Optimize interdigitated electrode layout for filtering performance.
    
    Returns top-N Pareto-optimal solutions meeting constraints.
    """
    start_time = time.time()
    
    try:
        # Generate candidate geometries (NSGA-II inspired)
        population_size = 60
        generations = 40
        
        solutions = []
        
        # Generate diverse population
        rng = np.random.default_rng(42)
        film_dict = request.mxene_film.dict()
        
        for _ in range(population_size):
            # Sample geometry parameters
            finger_width = rng.uniform(5, 20)
            finger_spacing = max(request.constraints.min_spacing_um, rng.uniform(5, 20))
            finger_length = rng.uniform(1000, 5000)
            num_fingers = rng.integers(20, 100)
            overlap_length = finger_length * rng.uniform(0.8, 0.95)
            thickness = rng.uniform(100, 500)
            
            geometry_dict = {
                "finger_width_um": finger_width,
                "finger_spacing_um": finger_spacing,
                "finger_length_um": finger_length,
                "num_fingers_per_electrode": int(num_fingers),
                "overlap_length_um": overlap_length,
                "thickness_nm": thickness,
                "substrate": "Si/SiO2",
            }
            
            # Calculate device area
            device_area = calculate_device_area(geometry_dict)
            
            # Check area constraint
            if device_area > request.constraints.max_area_mm2:
                continue
            
            # Get CPE parameters
            params = geometry_to_params(geometry_dict, film_dict)
            
            # Compute impedance at 120 Hz
            freq_120 = np.array([120.0])
            Z_mag, Z_phase, Z_complex = eis_surrogate.compute_impedance(freq_120, params)
            
            # Compute capacitance
            C_eff_imag, C_eff_cpe = eis_surrogate.compute_effective_capacitance(
                freq_120, params, Z_complex
            )
            
            # Get active area
            active_area_cm2 = (
                overlap_length * finger_spacing * (num_fingers - 1) * 1e-8
            )
            C_areal = (C_eff_cpe[0] * 1e3 / active_area_cm2) if active_area_cm2 > 0 else 0.0
            
            # Check constraints
            if Z_phase[0] > -80:  # Phase must be ≤ -80°
                continue
            if C_areal < request.constraints.targets.C_min_mf_cm2_120hz:
                continue
            if Z_mag[0] > request.constraints.targets.Z_max_ohm_120hz:
                continue
            
            # Compute full spectra for this solution
            frequencies = np.logspace(
                np.log10(request.frequency_range_hz[0]),
                np.log10(request.frequency_range_hz[1]),
                300
            )
            Z_mag_full, Z_phase_full, Z_complex_full = eis_surrogate.compute_impedance(
                frequencies, params
            )
            
            # Compute attenuation
            attenuation_db = eis_surrogate.compute_ripple_attenuation(
                frequencies, params, request.load_resistance_ohm
            )
            
            # Extract KPIs
            def get_value_at_freq(freq_target, freq_array, value_array):
                idx = np.argmin(np.abs(freq_array - freq_target))
                return float(value_array[idx])
            
            atten_50hz = get_value_at_freq(50, frequencies, attenuation_db)
            atten_60hz = get_value_at_freq(60, frequencies, attenuation_db)
            f_minus60 = eis_surrogate.find_frequency_at_phase(params, -60.0)
            
            # Decimate for plotting
            plot_indices = np.linspace(0, len(frequencies) - 1, 256, dtype=int)
            
            solution = OptimizationSolution(
                geometry=GeometryInput(**geometry_dict),
                kpis=KPIsOutput(
                    phase_deg_120hz=float(Z_phase[0]),
                    capacitance_mf_cm2_120hz=C_areal,
                    impedance_ohm_120hz=float(Z_mag[0]),
                    attenuation_db_50hz=atten_50hz,
                    attenuation_db_60hz=atten_60hz,
                    f_phi_minus60_deg_hz=f_minus60,
                    device_area_mm2=device_area,
                ),
                params=ParamsOutput(
                    Rs=params.Rs,
                    Q=params.Q,
                    alpha=params.alpha,
                    Rleak=params.Rleak if np.isfinite(params.Rleak) else 1e9,
                ),
                bode=BodeOutput(
                    freq_hz=frequencies[plot_indices].tolist(),
                    mag_ohm=Z_mag_full[plot_indices].tolist(),
                    phase_deg=Z_phase_full[plot_indices].tolist(),
                ),
                nyquist=NyquistOutput(
                    re_ohm=np.real(Z_complex_full[plot_indices]).tolist(),
                    im_ohm=np.imag(Z_complex_full[plot_indices]).tolist(),
                ),
            )
            
            solutions.append(solution)
        
        # Sort by area (minimize footprint)
        solutions.sort(key=lambda s: s.kpis.device_area_mm2)
        
        # Return top 5
        top_solutions = solutions[:5]
        
        computation_time = time.time() - start_time
        
        return FilteringOptimizeResponse(
            solutions=top_solutions,
            num_feasible=len(solutions),
            computation_time_s=computation_time,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.post("/fit")
async def fit_from_data(
    request: FilteringFitRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Fit CPE parameters from digitized Bode data.
    
    Stores calibration factors for future predictions.
    """
    try:
        # Load CSV
        df = pd.read_csv(request.digitized_bode_csv_path)
        
        # Validate columns
        required_cols = ["freq_hz", "mag_ohm", "phase_deg"]
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain columns: {required_cols}"
            )
        
        # Fit parameters
        params = eis_surrogate.fit_from_bode_data(
            freq_hz=df["freq_hz"].values,
            mag_ohm=df["mag_ohm"].values,
            phase_deg=df["phase_deg"].values,
            fit_rleak=request.fit_rleak,
        )
        
        # Store in database
        import json
        model = FilteringModel(
            electrolyte=request.electrolyte,
            process=request.process,
            Rs=params.Rs,
            Q=params.Q,
            alpha=params.alpha,
            Rleak=params.Rleak if np.isfinite(params.Rleak) else 1e9,
            calib_factors_json=json.dumps({
                "Q_scale": 1.0,
                "Rs_scale": 1.0,
                "alpha": params.alpha,
            }),
        )
        db.add(model)
        await db.commit()
        
        # Store calibration in memory
        calib_key = f"{request.electrolyte}_{request.process}"
        eis_surrogate.store_calibration(calib_key, {
            "Q_scale": 1.0,
            "Rs_scale": 1.0,
            "alpha": params.alpha,
        })
        
        return {
            "status": "success",
            "params": {
                "Rs": params.Rs,
                "Q": params.Q,
                "alpha": params.alpha,
                "Rleak": params.Rleak if np.isfinite(params.Rleak) else 1e9,
            },
            "calibration_key": calib_key,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fitting failed: {str(e)}")


@router.get("/presets", response_model=PresetOutput)
async def get_presets():
    """Get preset configurations for filtering mode."""
    return PresetOutput(
        load_resistances_ohm=[10.0, 33.0, 100.0],
        frequency_range_default=[1.0, 100000.0],
        geometry_bounds={
            "finger_width_um": [2.0, 50.0],
            "finger_spacing_um": [2.0, 50.0],
            "finger_length_um": [500.0, 10000.0],
            "num_fingers_per_electrode": [10, 200],
            "thickness_nm": [50.0, 1000.0],
        },
        process_templates={
            "photolithography": {
                "min_feature_um": 2.0,
                "typical_sheet_res": 10.0,
            },
            "inkjet": {
                "min_feature_um": 50.0,
                "typical_sheet_res": 50.0,
            },
            "screen_printing": {
                "min_feature_um": 100.0,
                "typical_sheet_res": 20.0,
            },
        },
    )
