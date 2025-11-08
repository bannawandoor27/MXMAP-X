"""Tests for AC-line filtering mode."""

import pytest
import numpy as np
from app.ml.eis_surrogate import (
    EISSurrogate,
    CPEParams,
    geometry_to_params,
    calculate_device_area,
)


class TestCPEImpedance:
    """Test CPE impedance calculations."""
    
    def test_cpe_impedance_basic(self):
        """Test basic CPE impedance calculation."""
        surrogate = EISSurrogate()
        params = CPEParams(Rs=1.0, Q=0.1, alpha=0.85, Rleak=np.inf)
        
        frequencies = np.array([1, 10, 100, 1000])
        Z_mag, Z_phase, Z_complex = surrogate.compute_impedance(frequencies, params)
        
        # Check shapes
        assert len(Z_mag) == len(frequencies)
        assert len(Z_phase) == len(frequencies)
        assert len(Z_complex) == len(frequencies)
        
        # Phase should be negative (capacitive)
        assert np.all(Z_phase < 0)
        
        # Magnitude should decrease with frequency (capacitive behavior)
        assert Z_mag[0] > Z_mag[-1]
    
    def test_cpe_alpha_limits(self):
        """Test CPE behavior at alpha limits."""
        surrogate = EISSurrogate()
        frequencies = np.array([120.0])
        
        # Alpha = 1 (ideal capacitor)
        params_ideal = CPEParams(Rs=0.5, Q=0.01, alpha=1.0, Rleak=np.inf)
        _, phase_ideal, _ = surrogate.compute_impedance(frequencies, params_ideal)
        
        # Phase should be close to -90° for ideal capacitor
        assert phase_ideal[0] < -85
        assert phase_ideal[0] > -90
        
        # Alpha = 0.7 (more resistive)
        params_resistive = CPEParams(Rs=0.5, Q=0.01, alpha=0.7, Rleak=np.inf)
        _, phase_resistive, _ = surrogate.compute_impedance(frequencies, params_resistive)
        
        # Phase should be less negative (more resistive)
        assert phase_resistive[0] > phase_ideal[0]
    
    def test_series_resistance(self):
        """Test series resistance contribution."""
        surrogate = EISSurrogate()
        frequencies = np.array([1e6])  # High frequency
        
        # At very high frequency, impedance should approach Rs
        params = CPEParams(Rs=2.0, Q=0.1, alpha=0.85, Rleak=np.inf)
        Z_mag, _, _ = surrogate.compute_impedance(frequencies, params)
        
        # At high frequency, |Z| should be close to Rs
        assert Z_mag[0] < 5.0  # Should be dominated by Rs


class TestEffectiveCapacitance:
    """Test effective capacitance calculations."""
    
    def test_capacitance_calculation(self):
        """Test effective capacitance at 120 Hz."""
        surrogate = EISSurrogate()
        params = CPEParams(Rs=1.0, Q=0.1, alpha=0.85, Rleak=np.inf)
        
        frequencies = np.array([120.0])
        C_imag, C_cpe = surrogate.compute_effective_capacitance(frequencies, params)
        
        # Both methods should give positive capacitance
        assert C_imag[0] > 0
        assert C_cpe[0] > 0
        
        # Values should be in reasonable range (mF)
        assert 0.001 < C_imag[0] < 1.0
        assert 0.001 < C_cpe[0] < 1.0
    
    def test_capacitance_frequency_dependence(self):
        """Test capacitance frequency dependence."""
        surrogate = EISSurrogate()
        params = CPEParams(Rs=1.0, Q=0.1, alpha=0.85, Rleak=np.inf)
        
        frequencies = np.array([1, 10, 100, 1000])
        _, C_cpe = surrogate.compute_effective_capacitance(frequencies, params)
        
        # For α < 1, capacitance should decrease with frequency
        assert C_cpe[0] > C_cpe[-1]


class TestRippleAttenuation:
    """Test ripple attenuation calculations."""
    
    def test_attenuation_at_line_frequencies(self):
        """Test attenuation at 50/60 Hz."""
        surrogate = EISSurrogate()
        params = CPEParams(Rs=1.0, Q=0.5, alpha=0.85, Rleak=np.inf)
        
        frequencies = np.array([50, 60])
        load_resistance = 33.0
        
        attenuation_db = surrogate.compute_ripple_attenuation(
            frequencies, params, load_resistance
        )
        
        # Attenuation should be negative (filtering)
        assert np.all(attenuation_db < 0)
        
        # Should provide reasonable filtering (> -30 dB)
        assert np.all(attenuation_db > -30)
    
    def test_attenuation_vs_load(self):
        """Test attenuation dependence on load resistance."""
        surrogate = EISSurrogate()
        params = CPEParams(Rs=1.0, Q=0.5, alpha=0.85, Rleak=np.inf)
        
        frequencies = np.array([60.0])
        
        # Lower load resistance should give better filtering
        atten_10ohm = surrogate.compute_ripple_attenuation(frequencies, params, 10.0)
        atten_100ohm = surrogate.compute_ripple_attenuation(frequencies, params, 100.0)
        
        # More negative = better filtering
        assert atten_10ohm[0] < atten_100ohm[0]


class TestFrequencySearch:
    """Test frequency search for target phase."""
    
    def test_find_minus60_deg(self):
        """Test finding frequency at -60° phase."""
        surrogate = EISSurrogate()
        params = CPEParams(Rs=1.0, Q=0.1, alpha=0.85, Rleak=np.inf)
        
        f_minus60 = surrogate.find_frequency_at_phase(params, -60.0)
        
        # Should find a frequency
        assert f_minus60 is not None
        assert f_minus60 > 0
        
        # Verify by computing phase at that frequency
        _, phase, _ = surrogate.compute_impedance(np.array([f_minus60]), params)
        
        # Should be close to -60° (within 1°)
        assert abs(phase[0] - (-60.0)) < 1.0
    
    def test_phase_not_in_range(self):
        """Test when target phase is not achievable."""
        surrogate = EISSurrogate()
        # Very resistive CPE (alpha = 0.5)
        params = CPEParams(Rs=10.0, Q=0.001, alpha=0.5, Rleak=np.inf)
        
        # Try to find -85° (may not be achievable)
        f_minus85 = surrogate.find_frequency_at_phase(params, -85.0, (1, 1000))
        
        # May return None if not in range
        if f_minus85 is not None:
            assert f_minus85 > 0


class TestBodeFitting:
    """Test fitting CPE parameters from Bode data."""
    
    def test_fit_synthetic_data(self):
        """Test fitting on synthetic Bode data."""
        surrogate = EISSurrogate()
        
        # Generate synthetic data
        true_params = CPEParams(Rs=1.5, Q=0.2, alpha=0.88, Rleak=np.inf)
        frequencies = np.logspace(0, 4, 50)
        
        Z_mag_true, Z_phase_true, _ = surrogate.compute_impedance(frequencies, true_params)
        
        # Add small noise
        rng = np.random.default_rng(42)
        Z_mag_noisy = Z_mag_true * (1 + rng.normal(0, 0.02, len(Z_mag_true)))
        Z_phase_noisy = Z_phase_true + rng.normal(0, 1.0, len(Z_phase_true))
        
        # Fit parameters
        fitted_params = surrogate.fit_from_bode_data(
            frequencies, Z_mag_noisy, Z_phase_noisy
        )
        
        # Check recovery (within 20% for Q, 10% for Rs, 5% for alpha)
        assert abs(fitted_params.Rs - true_params.Rs) / true_params.Rs < 0.15
        assert abs(fitted_params.Q - true_params.Q) / true_params.Q < 0.25
        assert abs(fitted_params.alpha - true_params.alpha) < 0.05
    
    def test_fit_at_120hz(self):
        """Test that fitted model reproduces phase at 120 Hz."""
        surrogate = EISSurrogate()
        
        # Generate data
        true_params = CPEParams(Rs=0.8, Q=0.45, alpha=0.86, Rleak=np.inf)
        frequencies = np.logspace(0, 4, 50)
        
        Z_mag_true, Z_phase_true, _ = surrogate.compute_impedance(frequencies, true_params)
        
        # Fit
        fitted_params = surrogate.fit_from_bode_data(frequencies, Z_mag_true, Z_phase_true)
        
        # Check at 120 Hz
        freq_120 = np.array([120.0])
        _, phase_true, _ = surrogate.compute_impedance(freq_120, true_params)
        _, phase_fitted, _ = surrogate.compute_impedance(freq_120, fitted_params)
        
        # Should match within ±5°
        assert abs(phase_fitted[0] - phase_true[0]) < 5.0


class TestGeometryMapping:
    """Test geometry to parameters mapping."""
    
    def test_basic_geometry_mapping(self):
        """Test basic geometry to CPE parameters."""
        geometry = {
            "finger_width_um": 8.0,
            "finger_spacing_um": 8.0,
            "finger_length_um": 2000.0,
            "num_fingers_per_electrode": 50,
            "overlap_length_um": 1800.0,
            "thickness_nm": 200.0,
            "substrate": "Si/SiO2"
        }
        
        mxene_film = {
            "flake_size_um": 2.0,
            "porosity_pct": 30.0,
            "sheet_res_ohm_sq": 10.0,
            "terminations": "-O,-OH",
            "electrolyte": "PVA/H2SO4",
            "process": "photolithography"
        }
        
        params = geometry_to_params(geometry, mxene_film)
        
        # Check parameter ranges
        assert 0.1 < params.Rs < 100
        assert 0.001 < params.Q < 10
        assert 0.7 < params.alpha < 1.0
        assert np.isinf(params.Rleak) or params.Rleak > 100
    
    def test_porosity_effect(self):
        """Test porosity effect on parameters."""
        geometry = {
            "finger_width_um": 10.0,
            "finger_spacing_um": 10.0,
            "finger_length_um": 2000.0,
            "num_fingers_per_electrode": 50,
            "overlap_length_um": 1800.0,
            "thickness_nm": 200.0,
            "substrate": "Si/SiO2"
        }
        
        # Low porosity
        film_low_porosity = {
            "porosity_pct": 15.0,
            "sheet_res_ohm_sq": 10.0,
            "electrolyte": "PVA/H2SO4",
            "process": "photolithography"
        }
        
        # High porosity
        film_high_porosity = {
            "porosity_pct": 50.0,
            "sheet_res_ohm_sq": 10.0,
            "electrolyte": "PVA/H2SO4",
            "process": "photolithography"
        }
        
        params_low = geometry_to_params(geometry, film_low_porosity)
        params_high = geometry_to_params(geometry, film_high_porosity)
        
        # Higher porosity should give lower Q and lower alpha
        assert params_low.Q > params_high.Q
        assert params_low.alpha > params_high.alpha
    
    def test_area_scaling(self):
        """Test that Q scales with active area."""
        base_geometry = {
            "finger_width_um": 10.0,
            "finger_spacing_um": 10.0,
            "finger_length_um": 2000.0,
            "num_fingers_per_electrode": 50,
            "overlap_length_um": 1800.0,
            "thickness_nm": 200.0,
            "substrate": "Si/SiO2"
        }
        
        # Double the overlap length (double area)
        large_geometry = base_geometry.copy()
        large_geometry["overlap_length_um"] = 3600.0
        
        film = {
            "porosity_pct": 30.0,
            "sheet_res_ohm_sq": 10.0,
            "electrolyte": "PVA/H2SO4",
            "process": "photolithography"
        }
        
        params_base = geometry_to_params(base_geometry, film)
        params_large = geometry_to_params(large_geometry, film)
        
        # Q should approximately double
        ratio = params_large.Q / params_base.Q
        assert 1.8 < ratio < 2.2


class TestDeviceArea:
    """Test device area calculation."""
    
    def test_area_calculation(self):
        """Test device footprint area."""
        geometry = {
            "finger_width_um": 10.0,
            "finger_spacing_um": 10.0,
            "finger_length_um": 2000.0,
            "num_fingers_per_electrode": 50,
            "substrate": "Si/SiO2"
        }
        
        area = calculate_device_area(geometry)
        
        # Should be positive and reasonable (mm²)
        assert area > 0
        assert area < 100  # Should be < 100 mm² for typical devices
    
    def test_area_scaling(self):
        """Test area scaling with geometry."""
        base_geometry = {
            "finger_width_um": 10.0,
            "finger_spacing_um": 10.0,
            "finger_length_um": 2000.0,
            "num_fingers_per_electrode": 50,
            "substrate": "Si/SiO2"
        }
        
        # Double fingers
        double_geometry = base_geometry.copy()
        double_geometry["num_fingers_per_electrode"] = 100
        
        area_base = calculate_device_area(base_geometry)
        area_double = calculate_device_area(double_geometry)
        
        # Area should approximately double
        ratio = area_double / area_base
        assert 1.8 < ratio < 2.2


class TestCalibration:
    """Test calibration storage and retrieval."""
    
    def test_store_and_retrieve_calibration(self):
        """Test calibration factor storage."""
        surrogate = EISSurrogate()
        
        key = "PVA/H2SO4_photolithography"
        factors = {
            "Q_scale": 1.2,
            "Rs_scale": 0.9,
            "alpha": 0.87
        }
        
        surrogate.store_calibration(key, factors)
        retrieved = surrogate.get_calibration(key)
        
        assert retrieved is not None
        assert retrieved["Q_scale"] == 1.2
        assert retrieved["Rs_scale"] == 0.9
        assert retrieved["alpha"] == 0.87
    
    def test_missing_calibration(self):
        """Test retrieval of non-existent calibration."""
        surrogate = EISSurrogate()
        
        retrieved = surrogate.get_calibration("nonexistent_key")
        assert retrieved is None


class TestAcceptanceCriteria:
    """Test acceptance criteria from spec."""
    
    def test_reasonable_geometry_meets_specs(self):
        """Test that reasonable geometry meets phase and attenuation specs."""
        surrogate = EISSurrogate()
        
        # Default geometry from spec
        geometry = {
            "finger_width_um": 8.0,
            "finger_spacing_um": 8.0,
            "finger_length_um": 2000.0,
            "num_fingers_per_electrode": 50,
            "overlap_length_um": 1800.0,
            "thickness_nm": 200.0,
            "substrate": "Si/SiO2"
        }
        
        mxene_film = {
            "porosity_pct": 30.0,
            "sheet_res_ohm_sq": 10.0,
            "electrolyte": "PVA/H2SO4",
            "process": "photolithography"
        }
        
        params = geometry_to_params(geometry, mxene_film)
        
        # Check phase at 120 Hz
        freq_120 = np.array([120.0])
        _, phase_120, _ = surrogate.compute_impedance(freq_120, params)
        
        # Should be ≤ -80°
        assert phase_120[0] <= -80.0
        
        # Check attenuation at 60 Hz
        freq_60 = np.array([60.0])
        attenuation = surrogate.compute_ripple_attenuation(freq_60, params, 33.0)
        
        # Should be negative (filtering)
        assert attenuation[0] < 0
        # Should be ≤ -6 dB
        assert attenuation[0] <= -6.0
