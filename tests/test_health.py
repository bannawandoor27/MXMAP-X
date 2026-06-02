"""Tests for health check endpoint."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient) -> None:
    """Test health check endpoint."""
    response = await client.get("/api/v1/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "version" in data
    assert "database" in data
    assert "model" in data
    assert data["status"] in ["healthy", "unhealthy"]


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient) -> None:
    """Test root endpoint returns web UI (HTML)."""
    response = await client.get("/")

    assert response.status_code == 200
    # Root now serves the web interface (HTML), not a JSON API response
    assert "text/html" in response.headers.get("content-type", "")
