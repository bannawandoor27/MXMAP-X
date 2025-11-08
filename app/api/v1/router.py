"""API v1 router configuration."""

from fastapi import APIRouter
from app.api.v1.endpoints import predictions, devices, models, advanced, filtering

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

api_router.include_router(
    advanced.router,
    tags=["advanced"],
)

api_router.include_router(
    filtering.router,
    prefix="/filtering",
    tags=["filtering"],
)
