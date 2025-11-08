"""Device data management endpoints."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_db
from app.models.database import Device
from app.models.schemas import (
    DeviceCreate,
    DeviceResponse,
    DeviceListResponse,
)
from app.core.exceptions import NotFoundError, DatabaseError

router = APIRouter()


@router.get(
    "/devices",
    response_model=DeviceListResponse,
    status_code=status.HTTP_200_OK,
    summary="List training devices",
    description="""
    Retrieve paginated list of training data devices.
    
    Supports filtering by:
    - MXene type
    - Electrolyte
    - Thickness range
    
    **Example Request:**
    ```
    GET /api/v1/devices?page=1&page_size=50&mxene_type=Ti3C2Tx
    ```
    """,
)
async def list_devices(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    mxene_type: Optional[str] = Query(None, description="Filter by MXene type"),
    electrolyte: Optional[str] = Query(None, description="Filter by electrolyte"),
    min_thickness: Optional[float] = Query(None, ge=0.5, description="Minimum thickness (μm)"),
    max_thickness: Optional[float] = Query(None, le=50.0, description="Maximum thickness (μm)"),
    db: AsyncSession = Depends(get_db),
) -> DeviceListResponse:
    """
    List devices with pagination and filtering.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page
        mxene_type: Optional MXene type filter
        electrolyte: Optional electrolyte filter
        min_thickness: Optional minimum thickness filter
        max_thickness: Optional maximum thickness filter
        db: Database session
        
    Returns:
        Paginated list of devices
    """
    try:
        # Build query with filters
        query = select(Device)
        
        if mxene_type:
            query = query.where(Device.mxene_type == mxene_type)
        if electrolyte:
            query = query.where(Device.electrolyte == electrolyte)
        if min_thickness:
            query = query.where(Device.thickness_um >= min_thickness)
        if max_thickness:
            query = query.where(Device.thickness_um <= max_thickness)
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar_one()
        
        # Apply pagination
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size).order_by(Device.created_at.desc())
        
        # Execute query
        result = await db.execute(query)
        devices = result.scalars().all()
        
        # Calculate total pages
        total_pages = (total + page_size - 1) // page_size
        
        return DeviceListResponse(
            devices=[DeviceResponse.model_validate(device) for device in devices],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": f"Database error: {str(e)}",
                "error_code": "DATABASE_ERROR",
            },
        )


@router.get(
    "/devices/{device_id}",
    response_model=DeviceResponse,
    status_code=status.HTTP_200_OK,
    summary="Get device details",
    description="""
    Retrieve detailed information for a specific device.
    
    **Example Request:**
    ```
    GET /api/v1/devices/42
    ```
    """,
    responses={
        200: {"description": "Device found"},
        404: {"description": "Device not found"},
    },
)
async def get_device(
    device_id: int,
    db: AsyncSession = Depends(get_db),
) -> DeviceResponse:
    """
    Get device by ID.
    
    Args:
        device_id: Device ID
        db: Database session
        
    Returns:
        Device details
        
    Raises:
        HTTPException: If device not found
    """
    try:
        query = select(Device).where(Device.id == device_id)
        result = await db.execute(query)
        device = result.scalar_one_or_none()
        
        if device is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": f"Device with ID {device_id} not found",
                    "error_code": "NOT_FOUND",
                },
            )
        
        return DeviceResponse.model_validate(device)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": f"Database error: {str(e)}",
                "error_code": "DATABASE_ERROR",
            },
        )


@router.post(
    "/devices",
    response_model=DeviceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add new training data",
    description="""
    Add a new device to the training dataset.
    
    Requires complete device composition and measured performance metrics.
    
    **Example Request:**
    ```json
    {
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
        "areal_capacitance_mf_cm2": 350.5,
        "esr_ohm": 2.5,
        "rate_capability_percent": 85.0,
        "cycle_life_cycles": 10000,
        "source": "DOI:10.1234/example",
        "notes": "Experimental data from lab"
    }
    ```
    """,
    responses={
        201: {"description": "Device created successfully"},
        422: {"description": "Validation error"},
        500: {"description": "Database error"},
    },
)
async def create_device(
    device_data: DeviceCreate,
    db: AsyncSession = Depends(get_db),
) -> DeviceResponse:
    """
    Create new device entry.
    
    Args:
        device_data: Device composition and performance data
        db: Database session
        
    Returns:
        Created device with ID
        
    Raises:
        HTTPException: If creation fails
    """
    try:
        # Create device instance
        device = Device(**device_data.model_dump())
        
        # Add to database
        db.add(device)
        await db.flush()
        await db.refresh(device)
        
        return DeviceResponse.model_validate(device)
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": f"Failed to create device: {str(e)}",
                "error_code": "DATABASE_ERROR",
            },
        )
