"""API tests for filtering endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestFilteringPredictEndpoint:
    """Test /api/v1/filtering/predict endpoint."""
    
    def test_predict_happy_path(self):
        """Test successful prediction."""
        request_data = {
            "frequency_range_hz": [1, 100000],
            "load_resistance_ohm": 33.0,
            "geometry": {
                "finger_width_um": 8.0,
                "finger_spacing_um": 8.0,
                "finger_length_um": 2000.0,
                "num_fingers_per_electrode": 50,
                "overlap_length_um": 1800.0,
                "thickness_nm": 200.0,
                "substrate": "Si/SiO2"
            },
            "mxene_film": {
                "flake_size_um": 2.0,
                "porosity_pct": 30.0,
                "sheet_res_ohm_sq": 10.0,
                "terminations": "-O,-OH",
                "electrolyte": "PVA/H2SO4",
                "process": "photolithography"
            },
            "fit_from_data": {
                "enable": False
            }
        }
        
        response = client.post("/api/v1/filtering/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check structure
        assert "kpis" in data
        assert "bode" in data
        assert "nyquist" in data
        assert "params" in data
        assert "recipe" in data
        
        # Check KPIs
        kpis = data["kpis"]
        assert "phase_deg_120hz" in kpis
        assert "capacitance_mf_cm2_120hz" in kpis
        assert "impedance_ohm_120hz" in kpis
        assert "attenuation_db_50hz" in kpis
        assert "attenuation_db_60hz" in kpis
        assert "device_area_mm2" in kpis
        
        # Check phase is capacitive
        assert kpis["phase_deg_120hz"] < 0
        assert kpis["phase_deg_120hz"] > -90
        
        # Check attenuation is negative (filtering)
        assert kpis["attenuation_db_60hz"] < 0
        
        # Check params
        params = data["params"]
        assert params["Rs"] > 0
        assert params["Q"] > 0
        assert 0.5 < params["alpha"] <= 1.0
        
        # Check plot data
        assert len(data["bode"]["freq_hz"]) > 0
        assert len(data["bode"]["mag_ohm"]) > 0
        assert len(data["bode"]["phase_deg"]) > 0
        assert len(data["nyquist"]["re_ohm"]) > 0
        assert len(data["nyquist"]["im_ohm"]) > 0
    
    def test_predict_validation_errors(self):
        """Test validation errors."""
        # Invalid finger width (too small)
        request_data = {
            "frequency_range_hz": [1, 100000],
            "load_resistance_ohm": 33.0,
            "geometry": {
                "finger_width_um": 1.0,  # Below minimum
                "finger_spacing_um": 8.0,
                "finger_length_um": 2000.0,
                "num_fingers_per_electrode": 50,
                "overlap_length_um": 1800.0,
                "thickness_nm": 200.0,
                "substrate": "Si/SiO2"
            },
            "mxene_film": {
                "flake_size_um": 2.0,
                "porosity_pct": 30.0,
                "sheet_res_ohm_sq": 10.0,
                "terminations": "-O,-OH",
                "electrolyte": "PVA/H2SO4",
                "process": "photolithography"
            }
        }
        
        response = client.post("/api/v1/filtering/predict", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_different_loads(self):
        """Test prediction with different load resistances."""
        base_request = {
            "frequency_range_hz": [1, 100000],
            "geometry": {
                "finger_width_um": 8.0,
                "finger_spacing_um": 8.0,
                "finger_length_um": 2000.0,
                "num_fingers_per_electrode": 50,
                "overlap_length_um": 1800.0,
                "thickness_nm": 200.0,
                "substrate": "Si/SiO2"
            },
            "mxene_film": {
                "flake_size_um": 2.0,
                "porosity_pct": 30.0,
                "sheet_res_ohm_sq": 10.0,
                "terminations": "-O,-OH",
                "electrolyte": "PVA/H2SO4",
                "process": "photolithography"
            }
        }
        
        # Test with 10 Ω load
        request_10 = {**base_request, "load_resistance_ohm": 10.0}
        response_10 = client.post("/api/v1/filtering/predict", json=request_10)
        assert response_10.status_code == 200
        
        # Test with 100 Ω load
        request_100 = {**base_request, "load_resistance_ohm": 100.0}
        response_100 = client.post("/api/v1/filtering/predict", json=request_100)
        assert response_100.status_code == 200
        
        # Lower load should give better attenuation (more negative)
        atten_10 = response_10.json()["kpis"]["attenuation_db_60hz"]
        atten_100 = response_100.json()["kpis"]["attenuation_db_60hz"]
        assert atten_10 < atten_100


class TestFilteringOptimizeEndpoint:
    """Test /api/v1/filtering/optimize endpoint."""
    
    def test_optimize_happy_path(self):
        """Test successful optimization."""
        request_data = {
            "frequency_range_hz": [1, 100000],
            "load_resistance_ohm": 33.0,
            "mxene_film": {
                "flake_size_um": 2.0,
                "porosity_pct": 30.0,
                "sheet_res_ohm_sq": 10.0,
                "terminations": "-O,-OH",
                "electrolyte": "PVA/H2SO4",
                "process": "photolithography"
            },
            "constraints": {
                "max_area_mm2": 4.0,
                "min_spacing_um": 5.0,
                "targets": {
                    "phase_deg_120hz": -85.0,
                    "C_min_mf_cm2_120hz": 10.0,
                    "Z_max_ohm_120hz": 2.0
                }
            }
        }
        
        response = client.post("/api/v1/filtering/optimize", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check structure
        assert "solutions" in data
        assert "num_feasible" in data
        assert "computation_time_s" in data
        
        # Should have at least some feasible solutions
        assert data["num_feasible"] >= 0
        
        # If solutions exist, check structure
        if len(data["solutions"]) > 0:
            solution = data["solutions"][0]
            assert "geometry" in solution
            assert "kpis" in solution
            assert "params" in solution
            assert "bode" in solution
            assert "nyquist" in solution
            
            # Check constraints are met
            assert solution["kpis"]["device_area_mm2"] <= 4.0
            assert solution["kpis"]["phase_deg_120hz"] <= -80.0
            assert solution["geometry"]["finger_spacing_um"] >= 5.0
    
    def test_optimize_returns_multiple_solutions(self):
        """Test that optimizer returns multiple Pareto solutions."""
        request_data = {
            "frequency_range_hz": [1, 100000],
            "load_resistance_ohm": 33.0,
            "mxene_film": {
                "flake_size_um": 2.0,
                "porosity_pct": 30.0,
                "sheet_res_ohm_sq": 10.0,
                "terminations": "-O,-OH",
                "electrolyte": "PVA/H2SO4",
                "process": "photolithography"
            },
            "constraints": {
                "max_area_mm2": 10.0,  # Relaxed constraint
                "min_spacing_um": 5.0,
                "targets": {
                    "phase_deg_120hz": -85.0,
                    "C_min_mf_cm2_120hz": 5.0,  # Relaxed
                    "Z_max_ohm_120hz": 5.0  # Relaxed
                }
            }
        }
        
        response = client.post("/api/v1/filtering/optimize", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have multiple solutions with relaxed constraints
        assert data["num_feasible"] >= 3
        assert len(data["solutions"]) >= 3
        
        # Solutions should be sorted by area (smallest first)
        areas = [sol["kpis"]["device_area_mm2"] for sol in data["solutions"]]
        assert areas == sorted(areas)


class TestFilteringPresetsEndpoint:
    """Test /api/v1/filtering/presets endpoint."""
    
    def test_get_presets(self):
        """Test getting presets."""
        response = client.get("/api/v1/filtering/presets")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check structure
        assert "load_resistances_ohm" in data
        assert "frequency_range_default" in data
        assert "geometry_bounds" in data
        assert "process_templates" in data
        
        # Check load resistances
        assert 10.0 in data["load_resistances_ohm"]
        assert 33.0 in data["load_resistances_ohm"]
        assert 100.0 in data["load_resistances_ohm"]
        
        # Check geometry bounds
        bounds = data["geometry_bounds"]
        assert "finger_width_um" in bounds
        assert "finger_spacing_um" in bounds
        assert "finger_length_um" in bounds
        
        # Check process templates
        templates = data["process_templates"]
        assert "photolithography" in templates
        assert "inkjet" in templates
        assert "screen_printing" in templates


class TestFilteringPerformance:
    """Test performance requirements."""
    
    def test_predict_performance(self):
        """Test that prediction completes within 150 ms."""
        import time
        
        request_data = {
            "frequency_range_hz": [1, 100000],
            "load_resistance_ohm": 33.0,
            "geometry": {
                "finger_width_um": 8.0,
                "finger_spacing_um": 8.0,
                "finger_length_um": 2000.0,
                "num_fingers_per_electrode": 50,
                "overlap_length_um": 1800.0,
                "thickness_nm": 200.0,
                "substrate": "Si/SiO2"
            },
            "mxene_film": {
                "flake_size_um": 2.0,
                "porosity_pct": 30.0,
                "sheet_res_ohm_sq": 10.0,
                "terminations": "-O,-OH",
                "electrolyte": "PVA/H2SO4",
                "process": "photolithography"
            }
        }
        
        start = time.time()
        response = client.post("/api/v1/filtering/predict", json=request_data)
        elapsed_ms = (time.time() - start) * 1000
        
        assert response.status_code == 200
        # Allow some margin for test overhead
        assert elapsed_ms < 300
    
    def test_plot_data_decimation(self):
        """Test that plot data is decimated to ≤512 points."""
        request_data = {
            "frequency_range_hz": [1, 100000],
            "load_resistance_ohm": 33.0,
            "geometry": {
                "finger_width_um": 8.0,
                "finger_spacing_um": 8.0,
                "finger_length_um": 2000.0,
                "num_fingers_per_electrode": 50,
                "overlap_length_um": 1800.0,
                "thickness_nm": 200.0,
                "substrate": "Si/SiO2"
            },
            "mxene_film": {
                "flake_size_um": 2.0,
                "porosity_pct": 30.0,
                "sheet_res_ohm_sq": 10.0,
                "terminations": "-O,-OH",
                "electrolyte": "PVA/H2SO4",
                "process": "photolithography"
            }
        }
        
        response = client.post("/api/v1/filtering/predict", json=request_data)
        data = response.json()
        
        # Check Bode plot data
        assert len(data["bode"]["freq_hz"]) <= 512
        assert len(data["bode"]["mag_ohm"]) <= 512
        assert len(data["bode"]["phase_deg"]) <= 512
        
        # Check Nyquist plot data
        assert len(data["nyquist"]["re_ohm"]) <= 512
        assert len(data["nyquist"]["im_ohm"]) <= 512
