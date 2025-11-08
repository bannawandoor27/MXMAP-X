"""Tests for printing/process-aware design."""

import pytest
import numpy as np
from app.ml.printing_surrogates import PrintingSurrogate


class TestInkWindow:
    """Test ink formulation window recommendations."""
    
    def test_screen_printing_window(self):
        """Test screen printing ink window."""
        surrogate = PrintingSurrogate()
        
        window = surrogate.get_ink_window(
            process="screen",
            mxene_type="Ti3C2Tx",
            flake_size_um=2.0,
            target_thickness_nm=200,
        )
        
        # Check ranges are non-empty
        assert window.solids_wt_pct[0] < window.solids_wt_pct[1]
        assert window.viscosity_mPas[0] < window.viscosity_mPas[1]
        
        # Screen printing should have high viscosity
        assert window.viscosity_mPas[0] >= 800
        
        # Check flake distribution
        assert window.flake_distribution_um["d10"] < window.flake_distribution_um["d50"]
        assert window.flake_distribution_um["d50"] < window.flake_distribution_um["d90"]
    
    def test_inkjet_window(self):
        """Test inkjet ink window."""
        surrogate = PrintingSurrogate()
        
        window = surrogate.get_ink_window(
            process="inkjet",
            mxene_type="Ti3C2Tx",
            flake_size_um=0.5,
            target_thickness_nm=100,
        )
        
        # Inkjet should have low viscosity
        assert window.viscosity_mPas[1] <= 20
        
        # Should have low solids
        assert window.solids_wt_pct[1] <= 3
    
    def test_large_flake_adjustment(self):
        """Test that large flakes reduce max solids."""
        surrogate = PrintingSurrogate()
        
        window_small = surrogate.get_ink_window(
            process="screen",
            mxene_type="Ti3C2Tx",
            flake_size_um=1.0,
            target_thickness_nm=200,
        )
        
        window_large = surrogate.get_ink_window(
            process="screen",
            mxene_type="Ti3C2Tx",
            flake_size_um=5.0,
            target_thickness_nm=200,
        )
        
        # Large flakes should have lower max solids
        assert window_large.solids_wt_pct[1] < window_small.solids_wt_pct[1]


class TestPrintedFilm:
    """Test printed film property estimation."""
    
    def test_basic_estimation(self):
        """Test basic film property estimation."""
        surrogate = PrintingSurrogate()
        
        film = surrogate.estimate_printed_film(
            process="screen",
            solids_wt_pct=8.0,
            viscosity_mPas=1500,
            flake_size_um=2.0,
            passes=2,
        )
        
        # Check properties are reasonable
        assert film.sheet_res_ohm_sq > 0
        assert 0 < film.transmittance_550nm <= 1.0
        assert film.haacke_FoM > 0
        assert film.thickness_nm > 0
        assert 0 <= film.coverage_factor <= 1.0
    
    def test_percolation_threshold(self):
        """Test percolation behavior."""
        surrogate = PrintingSurrogate()
        
        # Below percolation (1 pass, low solids)
        film_low = surrogate.estimate_printed_film(
            process="inkjet",
            solids_wt_pct=0.5,
            viscosity_mPas=10,
            flake_size_um=1.0,
            passes=1,
        )
        
        # Above percolation (multiple passes)
        film_high = surrogate.estimate_printed_film(
            process="inkjet",
            solids_wt_pct=2.0,
            viscosity_mPas=10,
            flake_size_um=1.0,
            passes=5,
        )
        
        # High coverage should have much lower resistance
        assert film_high.sheet_res_ohm_sq < film_low.sheet_res_ohm_sq * 0.1
    
    def test_thickness_scaling(self):
        """Test thickness scales with passes."""
        surrogate = PrintingSurrogate()
        
        film_1pass = surrogate.estimate_printed_film(
            process="screen",
            solids_wt_pct=8.0,
            viscosity_mPas=1500,
            flake_size_um=2.0,
            passes=1,
        )
        
        film_3pass = surrogate.estimate_printed_film(
            process="screen",
            solids_wt_pct=8.0,
            viscosity_mPas=1500,
            flake_size_um=2.0,
            passes=3,
        )
        
        # Thickness should scale roughly with passes
        ratio = film_3pass.thickness_nm / film_1pass.thickness_nm
        assert 2.5 < ratio < 3.5


class TestPostTreatment:
    """Test post-treatment effects."""
    
    def test_annealing_reduces_rs(self):
        """Test annealing reduces sheet resistance."""
        surrogate = PrintingSurrogate()
        
        effect = surrogate.estimate_post_treatment_effect(
            anneal_C=150,
            anneal_min=30,
            press_MPa=None,
            initial_Rs=50.0,
            initial_T=0.85,
        )
        
        # Annealing should reduce Rs
        assert effect.delta_rs_pct < 0
        assert effect.delta_rs_pct >= -40  # Max reduction
        
        # Should reduce roughness
        assert effect.roughness_change_nm < 0
    
    def test_pressing_reduces_rs(self):
        """Test pressing reduces sheet resistance."""
        surrogate = PrintingSurrogate()
        
        effect = surrogate.estimate_post_treatment_effect(
            anneal_C=None,
            anneal_min=None,
            press_MPa=10,
            initial_Rs=50.0,
            initial_T=0.85,
        )
        
        # Pressing should reduce Rs
        assert effect.delta_rs_pct < 0
        assert effect.delta_rs_pct >= -30
    
    def test_combined_treatment(self):
        """Test combined annealing and pressing."""
        surrogate = PrintingSurrogate()
        
        effect_combined = surrogate.estimate_post_treatment_effect(
            anneal_C=150,
            anneal_min=30,
            press_MPa=10,
            initial_Rs=50.0,
            initial_T=0.85,
        )
        
        effect_anneal = surrogate.estimate_post_treatment_effect(
            anneal_C=150,
            anneal_min=30,
            press_MPa=None,
            initial_Rs=50.0,
            initial_T=0.85,
        )
        
        # Combined should be better than anneal alone (synergy)
        assert effect_combined.delta_rs_pct < effect_anneal.delta_rs_pct


class TestRsTCurve:
    """Test Rs-T trade-off curve generation."""
    
    def test_curve_generation(self):
        """Test Rs-T curve generation."""
        surrogate = PrintingSurrogate()
        
        rs_curve, t_curve = surrogate.generate_rs_t_curve(
            process="screen",
            solids_wt_pct=8.0,
            viscosity_mPas=1500,
            flake_size_um=2.0,
            n_points=10,
        )
        
        # Check shapes
        assert len(rs_curve) == 10
        assert len(t_curve) == 10
        
        # Rs should decrease as T decreases (more passes)
        assert rs_curve[0] > rs_curve[-1]
        assert t_curve[0] > t_curve[-1]
    
    def test_monotonic_tradeoff(self):
        """Test Rs-T trade-off is monotonic."""
        surrogate = PrintingSurrogate()
        
        rs_curve, t_curve = surrogate.generate_rs_t_curve(
            process="screen",
            solids_wt_pct=8.0,
            viscosity_mPas=1500,
            flake_size_um=2.0,
            n_points=20,
        )
        
        # Rs should be monotonically decreasing
        assert np.all(np.diff(rs_curve) <= 0)
        
        # T should be monotonically decreasing
        assert np.all(np.diff(t_curve) <= 0)


class TestHaackeFoM:
    """Test Haacke Figure of Merit calculation."""
    
    def test_fom_calculation(self):
        """Test Haacke FoM calculation."""
        surrogate = PrintingSurrogate()
        
        film = surrogate.estimate_printed_film(
            process="screen",
            solids_wt_pct=8.0,
            viscosity_mPas=1500,
            flake_size_um=2.0,
            passes=2,
        )
        
        # Manually calculate FoM
        expected_fom = (film.transmittance_550nm ** 10) / film.sheet_res_ohm_sq
        
        # Should match
        assert abs(film.haacke_FoM - expected_fom) < 1e-6
    
    def test_fom_improves_with_treatment(self):
        """Test FoM improves with post-treatment."""
        surrogate = PrintingSurrogate()
        
        film = surrogate.estimate_printed_film(
            process="screen",
            solids_wt_pct=8.0,
            viscosity_mPas=1500,
            flake_size_um=2.0,
            passes=2,
        )
        
        effect = surrogate.estimate_post_treatment_effect(
            anneal_C=150,
            anneal_min=30,
            press_MPa=10,
            initial_Rs=film.sheet_res_ohm_sq,
            initial_T=film.transmittance_550nm,
        )
        
        # Apply treatment
        final_rs = film.sheet_res_ohm_sq * (1 + effect.delta_rs_pct / 100)
        final_t = film.transmittance_550nm * (1 + effect.delta_T_pct / 100)
        final_fom = (final_t ** 10) / final_rs
        
        # FoM should improve (Rs reduction dominates T reduction)
        assert final_fom > film.haacke_FoM


class TestManufacturabilityScore:
    """Test manufacturability score calculation."""
    
    def test_score_bounds(self):
        """Test score is bounded 0-100."""
        surrogate = PrintingSurrogate()
        
        score = surrogate.calculate_manufacturability_score(
            process="screen",
            solids_wt_pct=8.0,
            viscosity_mPas=1500,
            flake_d90_um=6.0,
            passes=2,
            anneal_C=120,
            press_MPa=10,
            substrate="PET",
        )
        
        assert 0 <= score <= 100
    
    def test_in_window_scores_high(self):
        """Test formulation in window scores high."""
        surrogate = PrintingSurrogate()
        
        # Get window
        window = surrogate.get_ink_window(
            process="screen",
            mxene_type="Ti3C2Tx",
            flake_size_um=2.0,
            target_thickness_nm=200,
        )
        
        # Use mid-range values
        mid_solids = np.mean(window.solids_wt_pct)
        mid_viscosity = np.mean(window.viscosity_mPas)
        
        score = surrogate.calculate_manufacturability_score(
            process="screen",
            solids_wt_pct=mid_solids,
            viscosity_mPas=mid_viscosity,
            flake_d90_um=6.0,
            passes=2,
            anneal_C=120,
            press_MPa=10,
            substrate="PET",
        )
        
        # Should score reasonably high
        assert score >= 70
    
    def test_out_of_window_penalized(self):
        """Test out-of-window formulation is penalized."""
        surrogate = PrintingSurrogate()
        
        # Way out of range for screen printing
        score = surrogate.calculate_manufacturability_score(
            process="screen",
            solids_wt_pct=20.0,  # Too high
            viscosity_mPas=100,  # Too low
            flake_d90_um=15.0,  # Too large
            passes=8,  # Too many
            anneal_C=250,  # Too high for PET
            press_MPa=30,  # Too high
            substrate="PET",
        )
        
        # Should score low
        assert score < 50
    
    def test_inkjet_clog_risk(self):
        """Test inkjet clog risk penalty."""
        surrogate = PrintingSurrogate()
        
        score_safe = surrogate.calculate_manufacturability_score(
            process="inkjet",
            solids_wt_pct=1.5,
            viscosity_mPas=10,
            flake_d90_um=0.8,  # Small flakes
            passes=3,
            anneal_C=100,
            press_MPa=None,
            substrate="PET",
        )
        
        score_risky = surrogate.calculate_manufacturability_score(
            process="inkjet",
            solids_wt_pct=1.5,
            viscosity_mPas=10,
            flake_d90_um=2.0,  # Large flakes - clog risk
            passes=3,
            anneal_C=100,
            press_MPa=None,
            substrate="PET",
        )
        
        # Risky should score lower
        assert score_risky < score_safe


class TestCalibration:
    """Test calibration from data."""
    
    def test_fit_from_data(self):
        """Test fitting from experimental data."""
        surrogate = PrintingSurrogate()
        
        # Generate synthetic data
        data = []
        for i in range(10):
            data.append({
                "process": "screen",
                "solids_wt_pct": 8.0 + i * 0.5,
                "viscosity_mPas": 1500,
                "flake_d50_um": 2.0,
                "passes": 2,
                "Rs_ohm_sq": 20.0 + i * 2,
                "T_550nm": 0.85 - i * 0.02,
            })
        
        metrics = surrogate.fit_from_data(data, "screen")
        
        # Check metrics exist
        assert "Rs_MAE_pct" in metrics
        assert "T_MAE_pct" in metrics
        assert "n_samples" in metrics
        
        assert metrics["n_samples"] == 10
    
    def test_calibration_storage(self):
        """Test calibration storage and retrieval."""
        surrogate = PrintingSurrogate()
        
        coeffs = {"rs_scale": 1.2, "t_scale": 0.95}
        metrics = {"Rs_MAE_pct": 12.5, "T_MAE_pct": 2.8}
        
        surrogate.store_calibration("screen", "v1.0", coeffs, metrics)
        
        retrieved = surrogate.get_calibration("screen")
        
        assert retrieved is not None
        assert retrieved["version"] == "v1.0"
        assert retrieved["coefficients"] == coeffs
        assert retrieved["metrics"] == metrics
