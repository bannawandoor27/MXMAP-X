"""API v1 router configuration."""

from fastapi import APIRouter
from app.api.v1.endpoints import predictions, devices, models

# Create v1 router
api_router = APIRouter()

# Include endpoint routers
api_router.include_router(
    predictions.router,
    tags=["predictions"],
)

api_router.include_router(
    devices.router,
    tags=["devices"],
)

api_router.include_router(
    models.router,
    tags=["models"],
)
