"""FastAPI application entry point."""

import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings
from app.api.v1.router import api_router
from app.core.exceptions import MXMAPException
from app.db.session import engine, Base


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup: Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield
    
    # Shutdown: Close database connections
    await engine.dispose()


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
    # MXMAP-X Backend API
    
    AI-powered MXene supercapacitor design tool for predicting device performance.
    
    ## Features
    
    - **ML Predictions**: Predict capacitance, ESR, rate capability, and cycle life
    - **Uncertainty Quantification**: 95% confidence intervals for all predictions
    - **Training Data Management**: Store and retrieve experimental device data
    - **Batch Processing**: Predict multiple devices in a single request
    - **Model Metrics**: Track ML model performance and versioning
    
    ## Quick Start
    
    1. **Health Check**: `GET /api/v1/health`
    2. **Single Prediction**: `POST /api/v1/predict`
    3. **Batch Prediction**: `POST /api/v1/predict/batch`
    4. **List Devices**: `GET /api/v1/devices`
    5. **Model Metrics**: `GET /api/v1/models/metrics`
    
    ## Authentication
    
    Currently no authentication required (development mode).
    Production deployment will require API keys.
    
    ## Rate Limits
    
    - Single predictions: No limit
    - Batch predictions: Max 100 devices per request
    
    ## Support
    
    For issues or questions, please contact the development team.
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    debug=settings.DEBUG,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


# Exception handlers
@app.exception_handler(MXMAPException)
async def mxmap_exception_handler(request: Request, exc: MXMAPException) -> JSONResponse:
    """Handle custom MXMAP exceptions."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": exc.message,
            "error_code": exc.error_code,
            "details": exc.details,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "error_code": "VALIDATION_ERROR",
            "details": exc.errors(),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "details": {"message": str(exc)} if settings.DEBUG else {},
            "request_id": getattr(request.state, "request_id", None),
        },
    )


# Include API router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

# Include web interface routes
from app.web.routes import router as web_router
app.include_router(web_router, tags=["web"])


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
