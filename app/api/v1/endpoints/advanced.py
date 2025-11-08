"""Advanced endpoints for optimization, exploration, and comparison."""

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import json

from app.core.dependencies import get_db
from app.models.database import Device
from app.models.schemas import PredictionRequest
from app.ml.model_loader import get_predictor
from app.ml.optimization import MultiObjectiveOptimizer, OptimizationObjective
from app.ml.chemistry_explorer import ChemistryExplorer
from pydantic import BaseModel, Field

router = APIRouter()


# ============================================================================
# Request/Response Schemas
# ============================================================================


class OptimizationRequest(BaseModel):
    """Request for multi-objective optimization."""
    
    objectives: list[dict[str, Any]] = Field(
        ...,
        description="List of objectives with metric, target, weight, constraints",
        examples=[[
            {"metric": "capacitance", "target": "maximize", "weight": 1.0},
            {"metric": "esr", "target": "minimize", "weight": 0.8},
            {"metric": "cycle_life", "target": "maximize", "weight": 0.6, "constraint_min": 8000}
        ]]
    )
    constraints: dict[str, Any] | None = Field(
        None,
        description="Design constraints (e.g., thickness_min, thickness_max)",
        examples=[{"thickness_min": 2.0, "thickness_max": 15.0}]
    )
    population_size: int = Field(
        100,
        ge=20,
        le=500,
        description="Population size for optimization"
    )
    generations: int = Field(
        50,
        ge=10,
        le=200,
        description="Number of optimization generations"
    )


class OptimizationResponse(BaseModel):
    """Response from multi-objective optimization."""
    
    pareto_optimal: list[dict[str, Any]] = Field(
        ...,
        description="Pareto-optimal solutions"
    )
    total_evaluated: int = Field(..., description="Total candidates evaluated")
    pareto_size: int = Field(..., description="Number of Pareto-optimal solutions")
    objectives: list[str] = Field(..., description="Optimization objectives")


class ChemistryMapRequest(BaseModel):
    """Request for chemistry space exploration."""
    
    n_samples: int = Field(
        200,
        ge=50,
        le=1000,
        description="Number of samples to generate"
    )
    n_neighbors: int = Field(
        15,
        ge=5,
        le=50,
        description="UMAP n_neighbors parameter"
    )
    min_dist: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="UMAP min_dist parameter"
    )


class CompareRequest(BaseModel):
    """Request to compare multiple candidates."""
    
    candidates: list[PredictionRequest] = Field(
        ...,
        min_length=2,
        max_length=10,
        description="List of candidates to compare (2-10)"
    )
    metrics: list[str] | None = Field(
        None,
        description="Metrics to compare (default: all)"
    )


class CompareResponse(BaseModel):
    """Response from comparison."""
    
    comparison: list[dict[str, Any]] = Field(..., description="Comparison results")
    rankings: dict[str, list[int]] = Field(..., description="Rankings by metric")
    best_overall: int = Field(..., description="Index of best overall candidate")
    summary: dict[str, Any] = Field(..., description="Summary statistics")


class RecipeCard(BaseModel):
    """Recipe card for device fabrication."""
    
    recipe_id: str = Field(..., description="Unique recipe identifier")
    device_composition: dict[str, Any] = Field(..., description="Material composition")
    processing_steps: list[dict[str, Any]] = Field(..., description="Fabrication steps")
    predicted_performance: dict[str, Any] = Field(..., description="Expected performance")
    materials_list: list[dict[str, Any]] = Field(..., description="Required materials")
    safety_notes: list[str] = Field(..., description="Safety considerations")
    estimated_time: str = Field(..., description="Total fabrication time")
    difficulty: str = Field(..., description="Difficulty level")


# ============================================================================
# Optimization Endpoint
# ============================================================================


@router.post(
    "/optimize",
    response_model=OptimizationResponse,
    status_code=status.HTTP_200_OK,
    summary="Multi-objective optimization",
    description="""
    Find Pareto-optimal device designs that balance multiple objectives.
    
    **Example Request:**
    ```json
    {
        "objectives": [
            {"metric": "capacitance", "target": "maximize", "weight": 1.0},
            {"metric": "esr", "target": "minimize", "weight": 0.8},
            {"metric": "cycle_life", "target": "maximize", "weight": 0.6}
        ],
        "constraints": {
            "thickness_min": 2.0,
            "thickness_max": 15.0
        },
        "population_size": 100,
        "generations": 50
    }
    ```
    
    Returns Pareto-optimal solutions that represent the best trade-offs
    between competing objectives.
    """,
)
async def optimize_design(request: OptimizationRequest) -> OptimizationResponse:
    """
    Perform multi-objective optimization.
    
    Args:
        request: Optimization parameters
        
    Returns:
        Pareto-optimal solutions
    """
    try:
        predictor = get_predictor()
        optimizer = MultiObjectiveOptimizer(predictor)
        
        # Parse objectives
        objectives = [
            OptimizationObjective(
                metric=obj["metric"],
                target=obj["target"],
                weight=obj.get("weight", 1.0),
                constraint_min=obj.get("constraint_min"),
                constraint_max=obj.get("constraint_max"),
            )
            for obj in request.objectives
        ]
        
        # Run optimization
        pareto_optimal = await optimizer.optimize(
            objectives=objectives,
            constraints=request.constraints,
            population_size=request.population_size,
            generations=request.generations,
        )
        
        # Format response
        pareto_solutions = [
            {
                "rank": candidate.rank,
                "composition": candidate.request.model_dump(),
                "predictions": candidate.predictions,
                "objectives": candidate.objectives,
                "crowding_distance": candidate.crowding_distance,
            }
            for candidate in pareto_optimal
        ]
        
        return OptimizationResponse(
            pareto_optimal=pareto_solutions,
            total_evaluated=request.population_size,
            pareto_size=len(pareto_optimal),
            objectives=[obj["metric"] for obj in request.objectives],
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": f"Optimization failed: {str(e)}",
                "error_code": "OPTIMIZATION_ERROR",
            },
        )


# ============================================================================
# Chemistry Exploration Endpoint
# ============================================================================


@router.get(
    "/explore",
    status_code=status.HTTP_200_OK,
    summary="Explore chemistry space",
    description="""
    Generate 2D chemistry map using UMAP dimensionality reduction.
    
    Visualizes the design space and identifies clusters of similar materials.
    
    **Query Parameters:**
    - `n_samples`: Number of samples to generate (50-1000)
    - `n_neighbors`: UMAP n_neighbors parameter (5-50)
    - `min_dist`: UMAP min_dist parameter (0.0-1.0)
    
    Returns 2D embeddings with predictions for visualization.
    """,
)
async def explore_chemistry_space(
    n_samples: int = 200,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Generate chemistry space map.
    
    Args:
        n_samples: Number of samples
        n_neighbors: UMAP parameter
        min_dist: UMAP parameter
        db: Database session
        
    Returns:
        Chemistry map data
    """
    try:
        predictor = get_predictor()
        explorer = ChemistryExplorer(predictor)
        
        # Try to get devices from database first
        query = select(Device).limit(n_samples)
        result = await db.execute(query)
        devices = result.scalars().all()
        
        device_dicts = []
        
        if len(devices) < 10:  # If not enough data, generate synthetic samples
            # Generate synthetic device compositions
            import numpy as np
            rng = np.random.default_rng(42)
            
            mxene_types = ["Ti3C2Tx", "Mo2CTx", "V2CTx", "Ti2CTx", "Nb2CTx"]
            terminations = ["O", "OH", "F", "mixed"]
            electrolytes = ["H2SO4", "KOH", "NaOH", "ionic_liquid", "PVA_H2SO4"]
            deposition_methods = ["vacuum_filtration", "spray_coating", "drop_casting", "spin_coating"]
            
            for i in range(n_samples):
                device_dicts.append({
                    "mxene_type": rng.choice(mxene_types),
                    "terminations": rng.choice(terminations),
                    "electrolyte": rng.choice(electrolytes),
                    "electrolyte_concentration": float(rng.uniform(0.5, 3.0)) if rng.random() > 0.3 else None,
                    "thickness_um": float(rng.uniform(1.0, 20.0)),
                    "deposition_method": rng.choice(deposition_methods),
                    "annealing_temp_c": float(rng.uniform(80, 200)) if rng.random() > 0.5 else None,
                    "annealing_time_min": float(rng.uniform(30, 120)) if rng.random() > 0.5 else None,
                    "interlayer_spacing_nm": float(rng.uniform(0.8, 1.5)) if rng.random() > 0.7 else None,
                    "specific_surface_area_m2g": float(rng.uniform(50, 300)) if rng.random() > 0.7 else None,
                    "pore_volume_cm3g": float(rng.uniform(0.1, 0.5)) if rng.random() > 0.8 else None,
                    "optical_transmittance": float(rng.uniform(0.3, 0.9)) if rng.random() > 0.8 else None,
                    "sheet_resistance_ohm_sq": float(rng.uniform(1, 100)) if rng.random() > 0.8 else None,
                })
        else:
            # Convert database devices to dict format
            device_dicts = [
                {
                    "mxene_type": d.mxene_type,
                    "terminations": d.terminations,
                    "electrolyte": d.electrolyte,
                    "electrolyte_concentration": d.electrolyte_concentration,
                    "thickness_um": d.thickness_um,
                    "deposition_method": d.deposition_method,
                    "annealing_temp_c": d.annealing_temp_c,
                    "annealing_time_min": d.annealing_time_min,
                    "interlayer_spacing_nm": d.interlayer_spacing_nm,
                    "specific_surface_area_m2g": d.specific_surface_area_m2g,
                    "pore_volume_cm3g": d.pore_volume_cm3g,
                    "optical_transmittance": d.optical_transmittance,
                    "sheet_resistance_ohm_sq": d.sheet_resistance_ohm_sq,
                }
                for d in devices
            ]
        
        # Generate map
        map_data = await explorer.generate_chemistry_map(
            devices=device_dicts,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
        
        return map_data
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": f"Chemistry exploration failed: {str(e)}",
                "error_code": "EXPLORATION_ERROR",
            },
        )


# ============================================================================
# Comparison Endpoint
# ============================================================================


@router.post(
    "/compare",
    response_model=CompareResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare multiple candidates",
    description="""
    Compare 2-10 device candidates side-by-side.
    
    **Example Request:**
    ```json
    {
        "candidates": [
            {
                "mxene_type": "Ti3C2Tx",
                "terminations": "O",
                "electrolyte": "H2SO4",
                "thickness_um": 5.0,
                "deposition_method": "vacuum_filtration"
            },
            {
                "mxene_type": "Mo2CTx",
                "terminations": "F",
                "electrolyte": "KOH",
                "thickness_um": 10.0,
                "deposition_method": "spray_coating"
            }
        ],
        "metrics": ["capacitance", "esr", "rate_capability"]
    }
    ```
    
    Returns predictions, rankings, and best overall candidate.
    """,
)
async def compare_candidates(request: CompareRequest) -> CompareResponse:
    """
    Compare multiple device candidates.
    
    Args:
        request: Comparison request
        
    Returns:
        Comparison results with rankings
    """
    try:
        predictor = get_predictor()
        
        # Get predictions for all candidates
        predictions = []
        for candidate in request.candidates:
            result = await predictor.predict(candidate)
            predictions.append({
                "composition": candidate.model_dump(),
                "capacitance": result.areal_capacitance.value,
                "esr": result.esr.value,
                "rate_capability": result.rate_capability.value,
                "cycle_life": result.cycle_life.value,
                "confidence": result.overall_confidence,
                "confidence_score": result.confidence_score,
            })
        
        # Calculate rankings for each metric
        metrics = request.metrics or ["capacitance", "esr", "rate_capability", "cycle_life"]
        rankings = {}
        
        for metric in metrics:
            values = [p[metric] for p in predictions]
            
            # Rank (lower is better for ESR, higher for others)
            if metric == "esr":
                ranked_indices = sorted(range(len(values)), key=lambda i: values[i])
            else:
                ranked_indices = sorted(range(len(values)), key=lambda i: -values[i])
            
            rankings[metric] = ranked_indices
        
        # Calculate best overall (average rank)
        avg_ranks = []
        for i in range(len(predictions)):
            ranks = [rankings[m].index(i) for m in metrics]
            avg_ranks.append(sum(ranks) / len(ranks))
        
        best_overall = int(avg_ranks.index(min(avg_ranks)))
        
        # Summary statistics
        summary = {
            "num_candidates": len(predictions),
            "metrics_compared": metrics,
            "best_capacitance_idx": rankings["capacitance"][0] if "capacitance" in rankings else None,
            "best_esr_idx": rankings["esr"][0] if "esr" in rankings else None,
            "best_rate_capability_idx": rankings["rate_capability"][0] if "rate_capability" in rankings else None,
            "best_cycle_life_idx": rankings["cycle_life"][0] if "cycle_life" in rankings else None,
        }
        
        return CompareResponse(
            comparison=predictions,
            rankings=rankings,
            best_overall=best_overall,
            summary=summary,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": f"Comparison failed: {str(e)}",
                "error_code": "COMPARISON_ERROR",
            },
        )


# ============================================================================
# Recipe Card Endpoint
# ============================================================================


@router.get(
    "/recipes/{device_id}",
    response_model=RecipeCard,
    status_code=status.HTTP_200_OK,
    summary="Export recipe card",
    description="""
    Generate a fabrication recipe card for a device.
    
    Includes:
    - Material composition
    - Step-by-step processing instructions
    - Predicted performance
    - Materials list
    - Safety notes
    - Estimated time and difficulty
    
    **Example:** `GET /api/v1/recipes/42`
    """,
)
async def get_recipe_card(
    device_id: int,
    db: AsyncSession = Depends(get_db),
) -> RecipeCard:
    """
    Generate recipe card for device fabrication.
    
    Args:
        device_id: Device ID
        db: Database session
        
    Returns:
        Recipe card with fabrication instructions
    """
    try:
        # Get device from database
        query = select(Device).where(Device.id == device_id)
        result = await db.execute(query)
        device = result.scalar_one_or_none()
        
        if device is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"Device {device_id} not found", "error_code": "NOT_FOUND"},
            )
        
        # Generate processing steps
        processing_steps = []
        
        # Step 1: MXene synthesis/preparation
        processing_steps.append({
            "step": 1,
            "title": "MXene Preparation",
            "description": f"Prepare {device.mxene_type} MXene with {device.terminations} terminations",
            "duration": "2-4 hours",
            "temperature": "Room temperature",
            "equipment": ["Sonicator", "Centrifuge", "Vacuum filtration setup"],
        })
        
        # Step 2: Electrolyte preparation
        if device.electrolyte_concentration:
            processing_steps.append({
                "step": 2,
                "title": "Electrolyte Preparation",
                "description": f"Prepare {device.electrolyte_concentration}M {device.electrolyte} solution",
                "duration": "30 minutes",
                "temperature": "Room temperature",
                "equipment": ["Volumetric flask", "Magnetic stirrer"],
            })
        
        # Step 3: Film deposition
        deposition_details = {
            "vacuum_filtration": {
                "description": f"Deposit MXene film via vacuum filtration to achieve {device.thickness_um}μm thickness",
                "duration": "1-2 hours",
                "equipment": ["Vacuum pump", "Filtration membrane", "Funnel"],
            },
            "spray_coating": {
                "description": f"Spray coat MXene dispersion to {device.thickness_um}μm thickness",
                "duration": "30-60 minutes",
                "equipment": ["Spray gun", "Hot plate", "Substrate holder"],
            },
            "drop_casting": {
                "description": f"Drop cast MXene dispersion to {device.thickness_um}μm thickness",
                "duration": "2-4 hours (including drying)",
                "equipment": ["Micropipette", "Substrate", "Drying oven"],
            },
        }
        
        deposition = deposition_details.get(device.deposition_method, deposition_details["vacuum_filtration"])
        processing_steps.append({
            "step": 3,
            "title": "Film Deposition",
            "description": deposition["description"],
            "duration": deposition["duration"],
            "temperature": "Room temperature",
            "equipment": deposition["equipment"],
        })
        
        # Step 4: Annealing (if applicable)
        if device.annealing_temp_c:
            processing_steps.append({
                "step": 4,
                "title": "Thermal Annealing",
                "description": f"Anneal at {device.annealing_temp_c}°C for {device.annealing_time_min} minutes",
                "duration": f"{device.annealing_time_min} minutes",
                "temperature": f"{device.annealing_temp_c}°C",
                "equipment": ["Tube furnace", "Inert gas supply"],
            })
        
        # Step 5: Device assembly
        processing_steps.append({
            "step": len(processing_steps) + 1,
            "title": "Device Assembly",
            "description": f"Assemble supercapacitor with {device.electrolyte} electrolyte",
            "duration": "1-2 hours",
            "temperature": "Room temperature",
            "equipment": ["Current collectors", "Separator", "Cell housing"],
        })
        
        # Materials list
        materials_list = [
            {"name": f"{device.mxene_type} MXene", "quantity": "~100 mg", "purity": "Research grade"},
            {"name": device.electrolyte, "quantity": "50-100 mL", "purity": "ACS grade"},
            {"name": "Deionized water", "quantity": "500 mL", "purity": "18.2 MΩ·cm"},
            {"name": "Current collectors", "quantity": "2 pieces", "purity": "N/A"},
            {"name": "Separator membrane", "quantity": "1 piece", "purity": "N/A"},
        ]
        
        # Safety notes
        safety_notes = [
            "Wear appropriate PPE (lab coat, gloves, safety glasses)",
            f"Handle {device.electrolyte} with care - corrosive material",
            "Work in well-ventilated area or fume hood",
            "Follow institutional chemical safety protocols",
        ]
        
        if device.annealing_temp_c and device.annealing_temp_c > 150:
            safety_notes.append("High temperature annealing - use heat-resistant gloves")
        
        # Estimated time and difficulty
        total_time_hours = sum([
            4,  # MXene prep
            0.5 if device.electrolyte_concentration else 0,  # Electrolyte
            2,  # Deposition
            (device.annealing_time_min / 60) if device.annealing_temp_c else 0,  # Annealing
            1.5,  # Assembly
        ])
        
        estimated_time = f"{total_time_hours:.1f} hours"
        
        # Difficulty based on complexity
        difficulty_score = 0
        if device.annealing_temp_c:
            difficulty_score += 1
        if device.deposition_method == "spray_coating":
            difficulty_score += 1
        if device.thickness_um < 2 or device.thickness_um > 20:
            difficulty_score += 1
        
        difficulty = ["Beginner", "Intermediate", "Advanced", "Expert"][min(difficulty_score, 3)]
        
        # Get predictions
        predictor = get_predictor()
        request = PredictionRequest(
            mxene_type=device.mxene_type,
            terminations=device.terminations,
            electrolyte=device.electrolyte,
            electrolyte_concentration=device.electrolyte_concentration,
            thickness_um=device.thickness_um,
            deposition_method=device.deposition_method,
            annealing_temp_c=device.annealing_temp_c,
            annealing_time_min=device.annealing_time_min,
            interlayer_spacing_nm=device.interlayer_spacing_nm,
            specific_surface_area_m2g=device.specific_surface_area_m2g,
            pore_volume_cm3g=device.pore_volume_cm3g,
            optical_transmittance=device.optical_transmittance,
            sheet_resistance_ohm_sq=device.sheet_resistance_ohm_sq,
        )
        
        prediction = await predictor.predict(request)
        
        return RecipeCard(
            recipe_id=f"MXMAP-{device_id:06d}",
            device_composition={
                "mxene_type": device.mxene_type,
                "terminations": device.terminations,
                "electrolyte": device.electrolyte,
                "thickness_um": device.thickness_um,
                "deposition_method": device.deposition_method,
            },
            processing_steps=processing_steps,
            predicted_performance={
                "areal_capacitance": f"{prediction.areal_capacitance.value} mF/cm²",
                "esr": f"{prediction.esr.value} Ω",
                "rate_capability": f"{prediction.rate_capability.value}%",
                "cycle_life": f"{prediction.cycle_life.value} cycles",
                "confidence": prediction.overall_confidence,
            },
            materials_list=materials_list,
            safety_notes=safety_notes,
            estimated_time=estimated_time,
            difficulty=difficulty,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": f"Recipe generation failed: {str(e)}",
                "error_code": "RECIPE_ERROR",
            },
        )


# ============================================================================
# WebSocket for Long-Running Predictions
# ============================================================================


@router.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for long-running predictions.
    
    Allows real-time progress updates for batch predictions
    and optimization tasks.
    
    **Message Format:**
    ```json
    {
        "type": "predict_batch",
        "data": {
            "devices": [...]
        }
    }
    ```
    
    **Response Format:**
    ```json
    {
        "type": "progress",
        "progress": 0.5,
        "message": "Processing device 50/100"
    }
    ```
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            
            if message_type == "predict_batch":
                # Handle batch prediction with progress updates
                devices_data = message.get("data", {}).get("devices", [])
                
                predictor = get_predictor()
                results = []
                
                total = len(devices_data)
                
                for i, device_data in enumerate(devices_data):
                    # Send progress update
                    await websocket.send_json({
                        "type": "progress",
                        "progress": (i + 1) / total,
                        "message": f"Processing device {i + 1}/{total}",
                    })
                    
                    # Make prediction
                    request = PredictionRequest(**device_data)
                    result = await predictor.predict(request)
                    
                    results.append({
                        "capacitance": result.areal_capacitance.value,
                        "esr": result.esr.value,
                        "rate_capability": result.rate_capability.value,
                        "cycle_life": result.cycle_life.value,
                        "confidence": result.overall_confidence,
                    })
                    
                    # Small delay to prevent overwhelming
                    await asyncio.sleep(0.01)
                
                # Send final results
                await websocket.send_json({
                    "type": "complete",
                    "results": results,
                })
            
            elif message_type == "ping":
                await websocket.send_json({"type": "pong"})
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}",
                })
    
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e),
        })
        await websocket.close()
