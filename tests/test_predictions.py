"""Tests for prediction endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_single_prediction(client: AsyncClient) -> None:
    """Test single device prediction."""
    request_data = {
        "mxene_type": "Ti3C2Tx",
        "terminations": "O",
        "electrolyte": "H2SO4",
        "electrolyte_concentration": 1.0,
        "thickness_um": 5.0,
        "deposition_method": "vacuum_filtration",
        "annealing_temp_c": 120.0,
        "annealing_time_min": 60.0,
        "interlayer_spacing_nm": 1.2,
        "specific_surface_area_m2g": 98.5,
    }
    
    response = await client.post("/api/v1/predict", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "areal_capacitance" in data
    assert "esr" in data
    assert "rate_capability" in data
    assert "cycle_life" in data
    assert "overall_confidence" in data
    assert "model_version" in data
    
    # Check uncertainty intervals
    cap = data["areal_capacitance"]
    assert "value" in cap
    assert "lower_ci" in cap
    assert "upper_ci" in cap
    assert "confidence" in cap
    assert cap["lower_ci"] < cap["value"] < cap["upper_ci"]


@pytest.mark.asyncio
async def test_batch_prediction(client: AsyncClient) -> None:
    """Test batch device prediction."""
    request_data = {
        "devices": [
            {
                "mxene_type": "Ti3C2Tx",
                "terminations": "O",
                "electrolyte": "H2SO4",
                "thickness_um": 5.0,
                "deposition_method": "vacuum_filtration",
            },
            {
                "mxene_type": "Mo2CTx",
                "terminations": "F",
                "electrolyte": "KOH",
                "thickness_um": 10.0,
                "deposition_method": "spray_coating",
            },
        ]
    }
    
    response = await client.post("/api/v1/predict/batch", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    assert data["total_count"] == 2
    assert data["successful_count"] == 2
    assert data["failed_count"] == 0


@pytest.mark.asyncio
async def test_prediction_validation_error(client: AsyncClient) -> None:
    """Test prediction with invalid input."""
    request_data = {
        "mxene_type": "InvalidType",  # Invalid enum value
        "terminations": "O",
        "electrolyte": "H2SO4",
        "thickness_um": 5.0,
        "deposition_method": "vacuum_filtration",
    }
    
    response = await client.post("/api/v1/predict", json=request_data)
    
    assert response.status_code == 422
