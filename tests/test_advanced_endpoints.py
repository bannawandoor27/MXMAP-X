"""Tests for advanced API endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_optimize_design(client: AsyncClient) -> None:
    """Test multi-objective optimization endpoint."""
    request_data = {
        "objectives": [
            {"metric": "capacitance", "target": "maximize", "weight": 1.0},
            {"metric": "esr", "target": "minimize", "weight": 0.8},
        ],
        "constraints": {
            "thickness_min": 2.0,
            "thickness_max": 15.0,
        },
        "population_size": 20,  # Small for testing
        "generations": 10,
    }
    
    response = await client.post("/api/v1/optimize", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "pareto_optimal" in data
    assert "total_evaluated" in data
    assert "pareto_size" in data
    assert len(data["pareto_optimal"]) > 0
    
    # Check Pareto solution structure
    solution = data["pareto_optimal"][0]
    assert "composition" in solution
    assert "predictions" in solution
    assert "objectives" in solution


@pytest.mark.asyncio
async def test_compare_candidates(client: AsyncClient) -> None:
    """Test candidate comparison endpoint."""
    request_data = {
        "candidates": [
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
        ],
        "metrics": ["capacitance", "esr"],
    }
    
    response = await client.post("/api/v1/compare", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "comparison" in data
    assert "rankings" in data
    assert "best_overall" in data
    assert "summary" in data
    
    assert len(data["comparison"]) == 2
    assert "capacitance" in data["rankings"]
    assert "esr" in data["rankings"]


@pytest.mark.asyncio
async def test_compare_validation_error(client: AsyncClient) -> None:
    """Test comparison with invalid input."""
    request_data = {
        "candidates": [
            {
                "mxene_type": "Ti3C2Tx",
                "terminations": "O",
                "electrolyte": "H2SO4",
                "thickness_um": 5.0,
                "deposition_method": "vacuum_filtration",
            },
        ],  # Only 1 candidate (minimum is 2)
    }
    
    response = await client.post("/api/v1/compare", json=request_data)
    
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_websocket_predict(client: AsyncClient) -> None:
    """Test WebSocket prediction endpoint."""
    # Note: This is a basic test. Full WebSocket testing requires more setup
    # In production, use a proper WebSocket testing library
    pass  # WebSocket testing requires special setup


@pytest.mark.asyncio
async def test_optimization_with_constraints(client: AsyncClient) -> None:
    """Test optimization with performance constraints."""
    request_data = {
        "objectives": [
            {
                "metric": "capacitance",
                "target": "maximize",
                "weight": 1.0,
                "constraint_min": 200.0,
            },
            {
                "metric": "cycle_life",
                "target": "maximize",
                "weight": 0.8,
                "constraint_min": 8000,
            },
        ],
        "population_size": 30,
        "generations": 10,
    }
    
    response = await client.post("/api/v1/optimize", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check that all solutions meet constraints
    for solution in data["pareto_optimal"]:
        assert solution["predictions"]["capacitance"] >= 200.0
        assert solution["predictions"]["cycle_life"] >= 8000
