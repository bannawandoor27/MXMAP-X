"""Printing process surrogates for ink formulation and film property prediction."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class InkWindow:
    """Recommended ink formulation window."""
    solids_wt_pct: Tuple[float, float]
    viscosity_mPas: Tuple[float, float]
    surface_tension_mNpm: Tuple[float, float]
    flake_distribution_um: Dict[str, float]
    additives: List[str]


@dataclass
class PrintedFilm:
    """Predicted printed film properties."""
    sheet_res_ohm_sq: float
    transmittance_550nm: float
    haacke_FoM: float
    thickness_nm: float
    coverage_factor: float


@dataclass
class PostTreatmentEffect:
    """Post-treatment effects on film properties."""
    delta_rs_pct: float
    delta_T_pct: float
    roughness_change_nm: float
    contact_improvement: float


class PrintingSurrogate:
    """
    Surrogate models for printing process design.
    
    Predicts ink formulation windows, printed film properties,
    and post-treatment effects for various printing processes.
    """
    
    # Process-specific parameters
    PROCESS_PARAMS = {
        "gravure": {
            "solids_range": (8, 15),
            "viscosity_range": (50, 200),
            "surface_tension": (30, 40),
            "typical_thickness_per_pass": 150,  # nm
            "coverage_efficiency": 0.92,
            "alignment_factor": 0.85,
        },
        "screen": {
            "solids_range": (5, 12),
            "viscosity_range": (800, 3000),
            "surface_tension": (30, 45),
            "typical_thickness_per_pass": 300,
            "coverage_efficiency": 0.88,
            "alignment_factor": 0.75,
        },
        "inkjet": {
            "solids_range": (0.5, 3),
            "viscosity_range": (5, 20),
            "surface_tension": (25, 35),
            "typical_thickness_per_pass": 50,
            "coverage_efficiency": 0.70,
            "alignment_factor": 0.60,
        },
        "slot-die": {
            "solids_range": (3, 10),
            "viscosity_range": (100, 500),
            "surface_tension": (28, 38),
            "typical_thickness_per_pass": 200,
            "coverage_efficiency": 0.90,
            "alignment_factor": 0.80,
        },
        "doctor-blade": {
            "solids_range": (5, 15),
            "viscosity_range": (200, 1000),
            "surface_tension": (30, 45),
            "typical_thickness_per_pass": 250,
            "coverage_efficiency": 0.85,
            "alignment_factor": 0.70,
        },
    }
    
    def __init__(self):
        """Initialize printing surrogate."""
        self.calibrations: Dict[str, Dict] = {}
    
    def get_ink_window(
        self,
        process: str,
        mxene_type: str,
        flake_size_um: float,
        target_thickness_nm: float,
    ) -> InkWindow:
        """
        Get recommended ink formulation window for a process.
        
        Args:
            process: Printing process
            mxene_type: MXene type
            flake_size_um: Average flake size
            target_thickness_nm: Target film thickness
            
        Returns:
            Ink formulation window
        """
        params = self.PROCESS_PARAMS.get(process, self.PROCESS_PARAMS["screen"])
        
        # Adjust ranges based on flake size
        flake_factor = flake_size_um / 2.0  # Normalize to 2 μm baseline
        
        solids_min, solids_max = params["solids_range"]
        visc_min, visc_max = params["viscosity_range"]
        
        # Larger flakes need lower solids to avoid clogging
        if flake_size_um > 3.0:
            solids_max *= 0.8
            visc_max *= 0.9
        
        # Adjust for target thickness
        passes_needed = target_thickness_nm / params["typical_thickness_per_pass"]
        if passes_needed > 3:
            # Multiple passes - can use lower solids
            solids_min *= 0.8
        
        # Flake size distribution (log-normal)
        d50 = flake_size_um
        d10 = d50 * 0.25
        d90 = d50 * 3.0
        
        # Additives based on process
        additives = []
        if process in ["inkjet", "slot-die"]:
            additives.append("surfactant<=0.2 wt%")
        if process in ["screen", "gravure"]:
            additives.append("binder<=0.5 wt%")
        if process == "inkjet":
            additives.append("humectant<=1.0 wt%")
        
        return InkWindow(
            solids_wt_pct=(solids_min, solids_max),
            viscosity_mPas=(visc_min, visc_max),
            surface_tension_mNpm=params["surface_tension"],
            flake_distribution_um={"d10": d10, "d50": d50, "d90": d90},
            additives=additives,
        )
    
    def estimate_printed_film(
        self,
        process: str,
        solids_wt_pct: float,
        viscosity_mPas: float,
        flake_size_um: float,
        passes: int,
        mxene_type: str = "Ti3C2Tx",
    ) -> PrintedFilm:
        """
        Estimate printed film properties.
        
        Args:
            process: Printing process
            solids_wt_pct: Ink solids content
            viscosity_mPas: Ink viscosity
            flake_size_um: Flake size
            passes: Number of printing passes
            mxene_type: MXene type
            
        Returns:
            Printed film properties
        """
        params = self.PROCESS_PARAMS.get(process, self.PROCESS_PARAMS["screen"])
        
        # Estimate thickness
        thickness_per_pass = params["typical_thickness_per_pass"]
        thickness_nm = thickness_per_pass * passes * (solids_wt_pct / 10.0)
        
        # Coverage factor (percolation model)
        coverage_efficiency = params["coverage_efficiency"]
        coverage_factor = min(1.0, coverage_efficiency * passes * 0.4)
        
        # Alignment factor (affects conductivity)
        alignment = params["alignment_factor"]
        
        # Percolation threshold
        percolation_threshold = 0.3
        if coverage_factor < percolation_threshold:
            # Below percolation - very high resistance
            sheet_res = 1e6
        else:
            # Effective medium approximation
            # Base conductivity for MXene
            base_conductivity = 5000  # S/cm for Ti3C2Tx
            
            # Effective conductivity scales with coverage and alignment
            percolation_factor = (coverage_factor - percolation_threshold) / (1 - percolation_threshold)
            effective_conductivity = base_conductivity * percolation_factor**2 * alignment
            
            # Sheet resistance = 1 / (conductivity × thickness)
            thickness_cm = thickness_nm * 1e-7
            sheet_res = 1.0 / (effective_conductivity * thickness_cm)
        
        # Transmittance (Beer-Lambert with scattering)
        # T = exp(-α × thickness) × (1 - haze)
        absorption_coeff = 0.003  # nm^-1
        haze_factor = 0.05 * (flake_size_um / 2.0)  # Larger flakes = more haze
        
        transmittance = np.exp(-absorption_coeff * thickness_nm) * (1 - haze_factor)
        transmittance = np.clip(transmittance, 0.1, 1.0)
        
        # Haacke Figure of Merit
        haacke_FoM = (transmittance ** 10) / sheet_res if sheet_res > 0 else 0
        
        return PrintedFilm(
            sheet_res_ohm_sq=sheet_res,
            transmittance_550nm=transmittance,
            haacke_FoM=haacke_FoM,
            thickness_nm=thickness_nm,
            coverage_factor=coverage_factor,
        )
    
    def estimate_post_treatment_effect(
        self,
        anneal_C: Optional[float],
        anneal_min: Optional[float],
        press_MPa: Optional[float],
        initial_Rs: float,
        initial_T: float,
    ) -> PostTreatmentEffect:
        """
        Estimate post-treatment effects on film properties.
        
        Args:
            anneal_C: Annealing temperature
            anneal_min: Annealing time
            press_MPa: Pressing pressure
            initial_Rs: Initial sheet resistance
            initial_T: Initial transmittance
            
        Returns:
            Post-treatment effects
        """
        delta_rs_pct = 0.0
        delta_T_pct = 0.0
        roughness_change_nm = 0.0
        contact_improvement = 1.0
        
        # Annealing effects
        if anneal_C and anneal_C > 80:
            # Mild annealing improves contact, reduces Rs
            anneal_factor = min((anneal_C - 80) / 120, 1.0)  # Saturates at 200°C
            time_factor = min((anneal_min or 30) / 60, 1.0) if anneal_min else 0.5
            
            # Rs reduction (exponential saturation)
            rs_reduction = -40 * anneal_factor * time_factor  # Up to -40%
            delta_rs_pct += rs_reduction
            
            # Slight T reduction due to densification
            delta_T_pct -= 2.0 * anneal_factor
            
            # Roughness reduction
            roughness_change_nm -= 10 * anneal_factor
            
            contact_improvement *= (1 + 0.5 * anneal_factor)
        
        # Pressing effects
        if press_MPa and press_MPa > 1:
            # Pressing improves flake-flake contact
            press_factor = min(press_MPa / 20, 1.0)  # Saturates at 20 MPa
            
            # Rs reduction from improved contact
            rs_reduction = -30 * press_factor  # Up to -30%
            delta_rs_pct += rs_reduction
            
            # Slight T reduction from densification
            delta_T_pct -= 1.5 * press_factor
            
            # Roughness reduction
            roughness_change_nm -= 15 * press_factor
            
            contact_improvement *= (1 + 0.4 * press_factor)
        
        # Combined effects (not simply additive due to saturation)
        if anneal_C and press_MPa:
            # Synergy bonus
            delta_rs_pct *= 1.1
        
        return PostTreatmentEffect(
            delta_rs_pct=delta_rs_pct,
            delta_T_pct=delta_T_pct,
            roughness_change_nm=roughness_change_nm,
            contact_improvement=contact_improvement,
        )
    
    def generate_rs_t_curve(
        self,
        process: str,
        solids_wt_pct: float,
        viscosity_mPas: float,
        flake_size_um: float,
        mxene_type: str = "Ti3C2Tx",
        n_points: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Rs-T trade-off curve by varying number of passes.
        
        Args:
            process: Printing process
            solids_wt_pct: Ink solids
            viscosity_mPas: Ink viscosity
            flake_size_um: Flake size
            mxene_type: MXene type
            n_points: Number of points on curve
            
        Returns:
            Tuple of (Rs array, T array)
        """
        passes_range = np.linspace(1, 10, n_points)
        rs_values = []
        t_values = []
        
        for passes in passes_range:
            film = self.estimate_printed_film(
                process, solids_wt_pct, viscosity_mPas,
                flake_size_um, int(passes), mxene_type
            )
            rs_values.append(film.sheet_res_ohm_sq)
            t_values.append(film.transmittance_550nm)
        
        return np.array(rs_values), np.array(t_values)
    
    def calculate_manufacturability_score(
        self,
        process: str,
        solids_wt_pct: float,
        viscosity_mPas: float,
        flake_d90_um: float,
        passes: int,
        anneal_C: Optional[float],
        press_MPa: Optional[float],
        substrate: str,
    ) -> int:
        """
        Calculate manufacturability score (0-100).
        
        Considers printability, risk, throughput, post-treatment burden, and yield.
        
        Args:
            process: Printing process
            solids_wt_pct: Ink solids
            viscosity_mPas: Ink viscosity
            flake_d90_um: 90th percentile flake size
            passes: Number of passes
            anneal_C: Annealing temperature
            press_MPa: Pressing pressure
            substrate: Substrate type
            
        Returns:
            Manufacturability score (0-100)
        """
        score = 100.0
        
        params = self.PROCESS_PARAMS.get(process, self.PROCESS_PARAMS["screen"])
        
        # 1. Printability window match (30 points)
        solids_min, solids_max = params["solids_range"]
        visc_min, visc_max = params["viscosity_range"]
        
        # Solids in range?
        if solids_min <= solids_wt_pct <= solids_max:
            score += 0  # No penalty
        else:
            deviation = min(abs(solids_wt_pct - solids_min), abs(solids_wt_pct - solids_max))
            score -= min(15, deviation * 3)
        
        # Viscosity in range?
        if visc_min <= viscosity_mPas <= visc_max:
            score += 0
        else:
            deviation_pct = min(
                abs(viscosity_mPas - visc_min) / visc_min,
                abs(viscosity_mPas - visc_max) / visc_max
            )
            score -= min(15, deviation_pct * 30)
        
        # 2. Risk factors (25 points)
        # Nozzle clog / mesh flooding risk
        if process == "inkjet":
            # Inkjet is sensitive to large flakes
            if flake_d90_um > 1.0:
                score -= 15
            if viscosity_mPas > 15:
                score -= 10
        elif process == "screen":
            # Screen can flood with low viscosity
            if viscosity_mPas < 1000:
                score -= 10
            if flake_d90_um > 10:
                score -= 5
        
        # 3. Throughput (20 points)
        # Fewer passes = better throughput
        if passes <= 2:
            score += 0
        elif passes <= 4:
            score -= 5
        else:
            score -= min(20, (passes - 4) * 3)
        
        # 4. Post-treatment burden (15 points)
        substrate_tg = {"PET": 120, "PI": 300, "glass": 500}.get(substrate, 150)
        
        if anneal_C:
            if anneal_C > substrate_tg:
                score -= 15  # Exceeds substrate limit
            elif anneal_C > substrate_tg * 0.8:
                score -= 8  # Close to limit
        
        if press_MPa and press_MPa > 15:
            score -= 5  # High pressure requirement
        
        # 5. Yield estimate (10 points)
        # Adhesion and cracking risk
        if substrate == "glass":
            score += 0  # Good adhesion
        elif substrate == "PET":
            if anneal_C and anneal_C > 100:
                score -= 3  # Risk of delamination
        
        return int(np.clip(score, 0, 100))
    
    def store_calibration(
        self,
        process: str,
        version: str,
        coefficients: Dict,
        metrics: Dict,
    ) -> None:
        """Store calibration for a process."""
        self.calibrations[process] = {
            "version": version,
            "coefficients": coefficients,
            "metrics": metrics,
        }
    
    def get_calibration(self, process: str) -> Optional[Dict]:
        """Get calibration for a process."""
        return self.calibrations.get(process)
    
    def fit_from_data(
        self,
        data: List[Dict],
        process: str,
    ) -> Dict:
        """
        Fit surrogate model from experimental data.
        
        Args:
            data: List of experimental measurements
            process: Process to calibrate
            
        Returns:
            Calibration metrics
        """
        # Simple calibration: compute scaling factors
        rs_errors = []
        t_errors = []
        
        for row in data:
            # Predict
            film = self.estimate_printed_film(
                process=row.get("process", process),
                solids_wt_pct=row["solids_wt_pct"],
                viscosity_mPas=row["viscosity_mPas"],
                flake_size_um=row["flake_d50_um"],
                passes=row.get("passes", 1),
            )
            
            # Compare with measured
            measured_rs = row["Rs_ohm_sq"]
            measured_t = row["T_550nm"]
            
            rs_error = abs(film.sheet_res_ohm_sq - measured_rs) / measured_rs
            t_error = abs(film.transmittance_550nm - measured_t) / measured_t
            
            rs_errors.append(rs_error)
            t_errors.append(t_error)
        
        # Compute metrics
        rs_mae_pct = np.mean(rs_errors) * 100
        t_mae_pct = np.mean(t_errors) * 100
        
        metrics = {
            "Rs_MAE_pct": rs_mae_pct,
            "T_MAE_pct": t_mae_pct,
            "n_samples": len(data),
        }
        
        # Store calibration (simplified - in production, fit scaling factors)
        coefficients = {
            "rs_scale": 1.0,
            "t_scale": 1.0,
        }
        
        self.store_calibration(process, "v1.0", coefficients, metrics)
        
        return metrics
