"""EIS surrogate model for AC-line filtering predictions."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class CPEParams:
    """Constant Phase Element parameters."""
    Rs: float  # Series resistance (Ω)
    Q: float  # CPE coefficient (F·s^(α-1))
    alpha: float  # CPE exponent (0 < α ≤ 1)
    Rleak: float  # Leakage resistance (Ω), can be inf


class EISSurrogate:
    """
    EIS surrogate model using Rs + (CPE || Rleak) circuit.
    
    Predicts impedance spectra, phase angles, and filtering performance
    for on-chip MXene MSCs in AC-line filtering applications.
    """

    def __init__(self):
        """Initialize EIS surrogate."""
        self.calibration_factors: Dict[str, Dict[str, float]] = {}

    def compute_impedance(
        self,
        frequencies: np.ndarray,
        params: CPEParams,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute complex impedance Z(jω) = Rs + (Z_CPE || R_leak).
        
        Args:
            frequencies: Frequency array in Hz
            params: CPE circuit parameters
            
        Returns:
            Tuple of (|Z| in Ω, phase in degrees, Z_complex)
        """
        omega = 2 * np.pi * frequencies
        j = 1j
        
        # CPE impedance: Z_CPE = 1 / (Q * (jω)^α)
        Z_cpe = 1.0 / (params.Q * np.power(j * omega, params.alpha))
        
        # Parallel combination with leakage resistance
        if np.isfinite(params.Rleak):
            Z_parallel = 1.0 / (1.0 / Z_cpe + 1.0 / params.Rleak)
        else:
            Z_parallel = Z_cpe
        
        # Total impedance
        Z_total = params.Rs + Z_parallel
        
        # Extract magnitude and phase
        Z_mag = np.abs(Z_total)
        Z_phase = np.angle(Z_total, deg=True)
        
        return Z_mag, Z_phase, Z_total

    def compute_effective_capacitance(
        self,
        frequencies: np.ndarray,
        params: CPEParams,
        Z_complex: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute effective capacitance at each frequency.
        
        Two methods:
        1. From imaginary impedance: C_eff = -1 / (2πf * Im(Z))
        2. From CPE: C_eff = (Q * ω^(α-1))^(1/α)
        
        Args:
            frequencies: Frequency array in Hz
            params: CPE parameters
            Z_complex: Optional pre-computed complex impedance
            
        Returns:
            Tuple of (C_eff_imag in F, C_eff_cpe in F)
        """
        omega = 2 * np.pi * frequencies
        
        # Method 1: From imaginary impedance
        if Z_complex is None:
            _, _, Z_complex = self.compute_impedance(frequencies, params)
        
        Z_imag = np.imag(Z_complex)
        # Avoid division by zero
        C_eff_imag = np.where(
            Z_imag != 0,
            -1.0 / (omega * Z_imag),
            0.0
        )
        
        # Method 2: From CPE formula
        C_eff_cpe = np.power(params.Q * np.power(omega, params.alpha - 1), 1.0 / params.alpha)
        
        return C_eff_imag, C_eff_cpe

    def compute_ripple_attenuation(
        self,
        frequencies: np.ndarray,
        params: CPEParams,
        load_resistance: float,
    ) -> np.ndarray:
        """
        Compute ripple attenuation when MSC is shunted across load.
        
        Transfer function: H(jω) = Z_MSC / (R_L + Z_MSC)
        Attenuation in dB: 20 * log10(|H|)
        
        Args:
            frequencies: Frequency array in Hz
            params: CPE parameters
            load_resistance: Load resistance in Ω
            
        Returns:
            Attenuation in dB (negative values indicate filtering)
        """
        _, _, Z_msc = self.compute_impedance(frequencies, params)
        
        # Transfer function
        H = Z_msc / (load_resistance + Z_msc)
        
        # Attenuation in dB
        attenuation_db = 20 * np.log10(np.abs(H))
        
        return attenuation_db

    def find_frequency_at_phase(
        self,
        params: CPEParams,
        target_phase_deg: float,
        freq_range: Tuple[float, float] = (1, 100000),
    ) -> Optional[float]:
        """
        Find frequency where phase angle equals target value.
        
        Args:
            params: CPE parameters
            target_phase_deg: Target phase angle in degrees
            freq_range: Search range (f_min, f_max) in Hz
            
        Returns:
            Frequency in Hz, or None if not found
        """
        # Sample frequencies logarithmically
        freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 1000)
        _, phases, _ = self.compute_impedance(freqs, params)
        
        # Find crossing point
        diff = phases - target_phase_deg
        
        # Check if target is within range
        if np.min(diff) > 0 or np.max(diff) < 0:
            return None
        
        # Find sign change
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_changes) == 0:
            return None
        
        # Linear interpolation at first crossing
        idx = sign_changes[0]
        f1, f2 = freqs[idx], freqs[idx + 1]
        p1, p2 = phases[idx], phases[idx + 1]
        
        # Interpolate
        f_target = f1 + (target_phase_deg - p1) * (f2 - f1) / (p2 - p1)
        
        return float(f_target)

    def fit_from_bode_data(
        self,
        freq_hz: np.ndarray,
        mag_ohm: np.ndarray,
        phase_deg: np.ndarray,
        initial_guess: Optional[CPEParams] = None,
        fit_rleak: bool = False,
    ) -> CPEParams:
        """
        Fit CPE parameters to measured Bode data.
        
        Args:
            freq_hz: Frequency array
            mag_ohm: Measured |Z| in Ω
            phase_deg: Measured phase in degrees
            initial_guess: Initial parameter guess
            fit_rleak: Whether to fit R_leak (otherwise set to inf)
            
        Returns:
            Fitted CPE parameters
        """
        if initial_guess is None:
            # Reasonable defaults for MXene MSCs
            initial_guess = CPEParams(Rs=1.0, Q=0.1, alpha=0.85, Rleak=np.inf)
        
        # Define objective function (weighted MSE)
        def objective(x):
            if fit_rleak:
                Rs, Q, alpha, Rleak = x
            else:
                Rs, Q, alpha = x
                Rleak = np.inf
            
            params = CPEParams(Rs=Rs, Q=Q, alpha=alpha, Rleak=Rleak)
            
            # Compute predicted impedance
            pred_mag, pred_phase, _ = self.compute_impedance(freq_hz, params)
            
            # Weighted MSE (log scale for magnitude, linear for phase)
            mag_error = np.mean((np.log10(pred_mag) - np.log10(mag_ohm)) ** 2)
            phase_error = np.mean((pred_phase - phase_deg) ** 2) / 100.0  # Scale down
            
            return mag_error + phase_error
        
        # Initial guess and bounds
        if fit_rleak:
            x0 = [initial_guess.Rs, initial_guess.Q, initial_guess.alpha, 1000.0]
            bounds = [(0.01, 100), (1e-6, 10), (0.5, 1.0), (10, 1e6)]
        else:
            x0 = [initial_guess.Rs, initial_guess.Q, initial_guess.alpha]
            bounds = [(0.01, 100), (1e-6, 10), (0.5, 1.0)]
        
        # Optimize
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        
        if fit_rleak:
            Rs, Q, alpha, Rleak = result.x
        else:
            Rs, Q, alpha = result.x
            Rleak = np.inf
        
        return CPEParams(Rs=Rs, Q=Q, alpha=alpha, Rleak=Rleak)

    def store_calibration(
        self,
        key: str,
        factors: Dict[str, float],
    ) -> None:
        """
        Store calibration factors for geometry→param scaling.
        
        Args:
            key: Calibration key (e.g., "PVA/H2SO4_photolithography")
            factors: Calibration factors dict
        """
        self.calibration_factors[key] = factors

    def get_calibration(self, key: str) -> Optional[Dict[str, float]]:
        """Get calibration factors for a key."""
        return self.calibration_factors.get(key)


def geometry_to_params(
    geometry: Dict,
    mxene_film: Dict,
    calibration: Optional[Dict[str, float]] = None,
) -> CPEParams:
    """
    Convert interdigitated electrode geometry to CPE parameters.
    
    Args:
        geometry: Geometry dict with finger dimensions
        mxene_film: MXene film properties
        calibration: Optional calibration factors
        
    Returns:
        CPE parameters
    """
    # Extract geometry
    finger_width_um = geometry["finger_width_um"]
    finger_spacing_um = geometry["finger_spacing_um"]
    finger_length_um = geometry["finger_length_um"]
    num_fingers = geometry["num_fingers_per_electrode"]
    overlap_length_um = geometry["overlap_length_um"]
    thickness_nm = geometry["thickness_nm"]
    
    # Extract film properties
    sheet_res = mxene_film.get("sheet_res_ohm_sq", 10.0)
    porosity_pct = mxene_film.get("porosity_pct", 30.0)
    
    # Calculate active area (overlap region between fingers)
    # Each pair of fingers creates an active region
    active_area_um2 = overlap_length_um * finger_spacing_um * (num_fingers - 1)
    active_area_cm2 = active_area_um2 * 1e-8
    
    # Series resistance: sheet resistance × (path length / width) + contact
    # Approximate current path as half the finger length
    path_length_um = finger_length_um / 2.0
    Rs_sheet = sheet_res * (path_length_um / finger_width_um) * 1e-4  # Convert to Ω
    Rs_contact = 0.5  # Typical contact resistance
    Rs = Rs_sheet + Rs_contact
    
    # CPE coefficient Q: scales with active area and porosity
    # Base capacitance ~10-20 mF/cm² for MXene MSCs
    base_cap_mf_cm2 = 15.0
    porosity_factor = 1.0 - (porosity_pct / 100.0)
    
    # Apply calibration if available
    if calibration:
        base_cap_mf_cm2 *= calibration.get("Q_scale", 1.0)
        Rs *= calibration.get("Rs_scale", 1.0)
    
    # Q in F·s^(α-1), assuming α ~ 0.85
    # For α=0.85, Q ≈ C_eff / ω^0.85 at reference frequency
    # Approximate: Q ≈ C_eff (at 1 Hz)
    C_eff_f = base_cap_mf_cm2 * active_area_cm2 * 1e-3  # Convert mF to F
    Q = C_eff_f * porosity_factor
    
    # CPE exponent: depends on porosity and roughness
    # Higher porosity → lower α (more deviation from ideal capacitor)
    alpha = 0.95 - (porosity_pct / 100.0) * 0.15
    alpha = np.clip(alpha, 0.75, 0.95)
    
    if calibration:
        alpha = calibration.get("alpha", alpha)
    
    # Leakage resistance: typically very high for solid/gel electrolytes
    Rleak = np.inf
    if "ionic_liquid" in mxene_film.get("electrolyte", "").lower():
        Rleak = 5000.0  # Lower for ionic liquids
    
    return CPEParams(Rs=Rs, Q=Q, alpha=alpha, Rleak=Rleak)


def calculate_device_area(geometry: Dict) -> float:
    """
    Calculate total device footprint area in mm².
    
    Args:
        geometry: Geometry dict
        
    Returns:
        Area in mm²
    """
    finger_width_um = geometry["finger_width_um"]
    finger_spacing_um = geometry["finger_spacing_um"]
    finger_length_um = geometry["finger_length_um"]
    num_fingers = geometry["num_fingers_per_electrode"]
    
    # Total width: (num_fingers * width) + ((num_fingers-1) * spacing)
    total_width_um = num_fingers * finger_width_um + (num_fingers - 1) * finger_spacing_um
    
    # Add margin for contacts and routing (20% on each side)
    device_width_um = total_width_um * 1.4
    device_length_um = finger_length_um * 1.2
    
    # Convert to mm²
    area_mm2 = (device_width_um * device_length_um) * 1e-6
    
    return area_mm2
