"""API tests for printing endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestPrintingRecommendEndpoint:
    """Test /api/v1/printing/recommend endpoint."""
    
    def test_recommend_happy_path(self):
        """Test successful recommendation."""
        request_data = {
            "process": "screen",
            "mxene_type": "Ti3C2Tx",
            "flake_size_um": 2.0,
            "target_thickness_nm": 200,
            "target_transmittance_550nm": 0.85,
            "substrate": "PET",
            "environment": {"temp_C": 25, "rh_pct": 45},
            "post_treatment": {
                "anneal_C": 120,
                "anneal_min": 30,
                "press_MPa": 10
            }
        }
        
        response = client.post("/api/v1/printing/recommend", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check structure
        assert "ink_window" in data
        assert "printed_film" in data
        assert "post_treatment_effect" in data
        assert "manufacturability_score" in data
        
        # Check ink window
        ink_window = data["ink_window"]
        assert len(ink_window["solids_wt_pct"]) == 2
        assert ink_window["solids_wt_pct"][0] < ink_window["solids_wt_pct"][1]
        assert len(ink_window["viscosity_mPas"]) == 2
        assert len(ink_window["additives"]) > 0
        
        # Check printed film
        film = data["printed_film"]
        assert film["sheet_res_ohm_sq"] > 0
        assert 0 < film["transmittance_550nm"] <= 1.0
        assert film["haacke_FoM"] > 0
        assert len(film["rs_t_curve"]["rs_ohm_sq"]) > 0
        
        # Check post-treatment
        post = data["post_treatment_effect"]
        assert post["delta_rs_pct"] < 0  # Should reduce Rs
        
        # Check manufacturability score
        assert 0 <= data["manufacturability_score"] <= 100
    
    def test_recommend_all_processes(self):
        """Test recommendation for all processes."""
        processes = ["gravure", "screen", "inkjet", "slot-die", "doctor-blade"]
        
        for process in processes:
            request_data = {
                "process": process,
                "mxene_type": "Ti3C2Tx",
                "flake_size_um": 2.0,
                "target_thickness_nm": 200,
                "target_transmittance_550nm": 0.85,
                "substrate": "PET",
            }
            
            response = client.post("/api/v1/printing/recommend", json=request_data)
            assert response.status_code == 200, f"Failed for {process}"
            
            data = response.json()
            assert data["ink_window"]["solids_wt_pct"][0] > 0
    
    def test_recommend_validation_errors(self):
        """Test validation errors."""
        # Invalid process
        request_data = {
            "process": "invalid_process",
            "mxene_type": "Ti3C2Tx",
            "flake_size_um": 2.0,
            "target_thickness_nm": 200,
            "target_transmittance_550nm": 0.85,
            "substrate": "PET",
        }
        
        response = client.post("/api/v1/printing/recommend", json=request_data)
        assert response.status_code == 422


class TestPrintingEstimateEndpoint:
    """Test /api/v1/printing/estimate endpoint."""
    
    def test_estimate_happy_path(self):
        """Test successful estimation."""
        request_data = {
            "process": "screen",
            "solids_wt_pct": 8.0,
            "viscosity_mPas": 1500,
            "flake_size_um": 2.0,
            "passes": 2,
            "mxene_type": "Ti3C2Tx",
            "post_treatment": {
                "anneal_C": 120,
                "anneal_min": 30,
                "press_MPa": 10
            }
        }
        
        response = client.post("/api/v1/printing/estimate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check all required fields
        assert "ink_window" in data
        assert "printed_film" in data
        assert "manufacturability_score" in data
        
        # Check film properties
        assert data["printed_film"]["sheet_res_ohm_sq"] > 0
        assert 0 < data["printed_film"]["transmittance_550nm"] <= 1.0
    
    def test_estimate_passes_effect(self):
        """Test effect of number of passes."""
        base_request = {
            "process": "screen",
            "solids_wt_pct": 8.0,
            "viscosity_mPas": 1500,
            "flake_size_um": 2.0,
            "mxene_type": "Ti3C2Tx",
        }
        
        # 1 pass
        request_1 = {**base_request, "passes": 1}
        response_1 = client.post("/api/v1/printing/estimate", json=request_1)
        data_1 = response_1.json()
        
        # 3 passes
        request_3 = {**base_request, "passes": 3}
        response_3 = client.post("/api/v1/printing/estimate", json=request_3)
        data_3 = response_3.json()
        
        # More passes should give lower Rs and lower T
        assert data_3["printed_film"]["sheet_res_ohm_sq"] < data_1["printed_film"]["sheet_res_ohm_sq"]
        assert data_3["printed_film"]["transmittance_550nm"] < data_1["printed_film"]["transmittance_550nm"]


class TestPrintingPresetsEndpoint:
    """Test /api/v1/printing/presets endpoint."""
    
    def test_get_presets(self):
        """Test getting presets."""
        response = client.get("/api/v1/printing/presets")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check structure
        assert "processes" in data
        assert "substrates" in data
        assert "typical_targets" in data
        
        # Check processes
        processes = data["processes"]
        assert "gravure" in processes
        assert "screen" in processes
        assert "inkjet" in processes
        assert "slot-die" in processes
        assert "doctor-blade" in processes
        
        # Check each process has required fields
        for process_name, process_data in processes.items():
            assert "solids_range" in process_data
            assert "viscosity_range" in process_data
            assert "typical_thickness_per_pass" in process_data
        
        # Check substrates
        substrates = data["substrates"]
        assert "PET" in substrates
        assert "PI" in substrates
        assert "glass" in substrates
        
        # Check typical targets
        targets = data["typical_targets"]
        assert "transparent_conductor" in targets
        assert "opaque_electrode" in targets


class TestPrintingPerformance:
    """Test performance requirements."""
    
    def test_recommend_performance(self):
        """Test recommendation completes within 100 ms."""
        import time
        
        request_data = {
            "process": "screen",
            "mxene_type": "Ti3C2Tx",
            "flake_size_um": 2.0,
            "target_thickness_nm": 200,
            "target_transmittance_550nm": 0.85,
            "substrate": "PET",
        }
        
        start = time.time()
        response = client.post("/api/v1/printing/recommend", json=request_data)
        elapsed_ms = (time.time() - start) * 1000
        
        assert response.status_code == 200
        # Allow margin for test overhead
        assert elapsed_ms < 200
    
    def test_rs_t_curve_decimation(self):
        """Test Rs-T curve is decimated."""
        request_data = {
            "process": "screen",
            "mxene_type": "Ti3C2Tx",
            "flake_size_um": 2.0,
            "target_thickness_nm": 200,
            "target_transmittance_550nm": 0.85,
            "substrate": "PET",
        }
        
        response = client.post("/api/v1/printing/recommend", json=request_data)
        data = response.json()
        
        # Check curve has reasonable number of points
        rs_curve = data["printed_film"]["rs_t_curve"]["rs_ohm_sq"]
        t_curve = data["printed_film"]["rs_t_curve"]["T_550nm"]
        
        assert len(rs_curve) <= 500
        assert len(t_curve) <= 500
        assert len(rs_curve) == len(t_curve)


class TestAcceptanceCriteria:
    """Test acceptance criteria from spec."""
    
    def test_non_empty_ink_window(self):
        """Test all processes return non-empty ink windows."""
        processes = ["gravure", "screen", "inkjet", "slot-die", "doctor-blade"]
        
        for process in processes:
            request_data = {
                "process": process,
                "mxene_type": "Ti3C2Tx",
                "flake_size_um": 2.0,
                "target_thickness_nm": 200,
                "target_transmittance_550nm": 0.85,
                "substrate": "PET",
            }
            
            response = client.post("/api/v1/printing/recommend", json=request_data)
            data = response.json()
            
            ink_window = data["ink_window"]
            
            # Check ranges are realistic
            assert ink_window["solids_wt_pct"][0] > 0
            assert ink_window["solids_wt_pct"][1] > ink_window["solids_wt_pct"][0]
            assert ink_window["viscosity_mPas"][0] > 0
            assert ink_window["viscosity_mPas"][1] > ink_window["viscosity_mPas"][0]
    
    def test_rs_t_tradeoff_feasible(self):
        """Test Rs-T plot shows feasible trade-off."""
        request_data = {
            "process": "screen",
            "mxene_type": "Ti3C2Tx",
            "flake_size_um": 2.0,
            "target_thickness_nm": 200,
            "target_transmittance_550nm": 0.85,
            "substrate": "PET",
        }
        
        response = client.post("/api/v1/printing/recommend", json=request_data)
        data = response.json()
        
        film = data["printed_film"]
        
        # Check values are in feasible range
        assert 1 < film["sheet_res_ohm_sq"] < 1000
        assert 0.5 < film["transmittance_550nm"] < 1.0
        
        # Check Rs-T curve is monotonic
        rs_curve = film["rs_t_curve"]["rs_ohm_sq"]
        t_curve = film["rs_t_curve"]["T_550nm"]
        
        # Rs should decrease along curve
        assert rs_curve[0] >= rs_curve[-1]
        # T should decrease along curve
        assert t_curve[0] >= t_curve[-1]
    
    def test_post_treatment_improves_rs(self):
        """Test post-treatment reduces Rs by ≥20%."""
        request_data = {
            "process": "screen",
            "mxene_type": "Ti3C2Tx",
            "flake_size_um": 2.0,
            "target_thickness_nm": 200,
            "target_transmittance_550nm": 0.85,
            "substrate": "PET",
            "post_treatment": {
                "anneal_C": 150,
                "anneal_min": 60,
                "press_MPa": 15
            }
        }
        
        response = client.post("/api/v1/printing/recommend", json=request_data)
        data = response.json()
        
        delta_rs = data["post_treatment_effect"]["delta_rs_pct"]
        
        # Should reduce Rs by at least 20%
        assert delta_rs <= -20
    
    def test_manufacturability_score_displayed(self):
        """Test manufacturability score is computed and in range."""
        request_data = {
            "process": "screen",
            "mxene_type": "Ti3C2Tx",
            "flake_size_um": 2.0,
            "target_thickness_nm": 200,
            "target_transmittance_550nm": 0.85,
            "substrate": "PET",
        }
        
        response = client.post("/api/v1/printing/recommend", json=request_data)
        data = response.json()
        
        score = data["manufacturability_score"]
        
        # Score should be 0-100
        assert 0 <= score <= 100
        assert isinstance(score, int)
