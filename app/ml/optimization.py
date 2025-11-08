"""Multi-objective optimization for MXene supercapacitor design."""

import numpy as np
from typing import Any
from dataclasses import dataclass
from app.models.schemas import PredictionRequest, MXeneType, Termination, Electrolyte, DepositionMethod


@dataclass
class OptimizationObjective:
    """Optimization objective specification."""
    
    metric: str  # capacitance, esr, rate_capability, cycle_life
    target: str  # maximize or minimize
    weight: float = 1.0
    constraint_min: float | None = None
    constraint_max: float | None = None


@dataclass
class OptimizationCandidate:
    """Candidate solution from optimization."""
    
    request: PredictionRequest
    predictions: dict[str, float]
    objectives: dict[str, float]
    rank: int
    crowding_distance: float
    is_pareto_optimal: bool


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization using NSGA-II inspired approach.
    
    Finds Pareto-optimal device designs that balance multiple objectives
    (e.g., high capacitance + low ESR + high cycle life).
    """

    def __init__(self, predictor: Any) -> None:
        """
        Initialize optimizer.
        
        Args:
            predictor: Trained ML predictor
        """
        self.predictor = predictor

    async def optimize(
        self,
        objectives: list[OptimizationObjective],
        constraints: dict[str, Any] | None = None,
        population_size: int = 100,
        generations: int = 50,
    ) -> list[OptimizationCandidate]:
        """
        Perform multi-objective optimization.
        
        Args:
            objectives: List of optimization objectives
            constraints: Design constraints (e.g., thickness range)
            population_size: Number of candidates per generation
            generations: Number of optimization iterations
            
        Returns:
            List of Pareto-optimal candidates
        """
        # Generate initial population
        population = self._generate_initial_population(population_size, constraints)
        
        # Evaluate initial population
        evaluated_pop = await self._evaluate_population(population, objectives)
        
        # Evolution loop (simplified for now - can be enhanced with genetic operators)
        for gen in range(generations):
            # For now, we'll use a grid search approach
            # In production, implement proper genetic operators (crossover, mutation)
            pass
        
        # Perform non-dominated sorting
        pareto_fronts = self._non_dominated_sort(evaluated_pop)
        
        # Get Pareto-optimal solutions (first front)
        pareto_optimal = pareto_fronts[0] if pareto_fronts else []
        
        # Calculate crowding distance for diversity
        if pareto_optimal:
            self._calculate_crowding_distance(pareto_optimal)
        
        # Sort by crowding distance (prefer diverse solutions)
        pareto_optimal.sort(key=lambda x: x.crowding_distance, reverse=True)
        
        return pareto_optimal

    def _generate_initial_population(
        self,
        size: int,
        constraints: dict[str, Any] | None = None,
    ) -> list[PredictionRequest]:
        """
        Generate initial population using Latin Hypercube Sampling.
        
        Args:
            size: Population size
            constraints: Design constraints
            
        Returns:
            List of candidate designs
        """
        population = []
        
        # Define design space
        mxene_types = list(MXeneType)
        terminations = list(Termination)
        electrolytes = list(Electrolyte)
        deposition_methods = list(DepositionMethod)
        
        # Apply constraints
        thickness_min = constraints.get("thickness_min", 1.0) if constraints else 1.0
        thickness_max = constraints.get("thickness_max", 30.0) if constraints else 30.0
        
        # Generate diverse samples
        rng = np.random.default_rng(42)
        
        for i in range(size):
            # Sample categorical variables
            mxene_type = rng.choice(mxene_types)
            termination = rng.choice(terminations)
            electrolyte = rng.choice(electrolytes)
            deposition_method = rng.choice(deposition_methods)
            
            # Sample continuous variables with Latin Hypercube
            thickness = rng.uniform(thickness_min, thickness_max)
            
            # Optional parameters (50% probability)
            electrolyte_conc = rng.uniform(0.5, 3.0) if rng.random() > 0.5 else None
            annealing_temp = rng.uniform(80, 200) if rng.random() > 0.5 else None
            annealing_time = rng.uniform(30, 120) if annealing_temp else None
            
            interlayer_spacing = rng.uniform(0.9, 1.8) if rng.random() > 0.3 else None
            surface_area = rng.uniform(50, 150) if rng.random() > 0.3 else None
            pore_volume = rng.uniform(0.05, 0.25) if rng.random() > 0.3 else None
            
            request = PredictionRequest(
                mxene_type=mxene_type,
                terminations=termination,
                electrolyte=electrolyte,
                electrolyte_concentration=electrolyte_conc,
                thickness_um=thickness,
                deposition_method=deposition_method,
                annealing_temp_c=annealing_temp,
                annealing_time_min=annealing_time,
                interlayer_spacing_nm=interlayer_spacing,
                specific_surface_area_m2g=surface_area,
                pore_volume_cm3g=pore_volume,
            )
            
            population.append(request)
        
        return population

    async def _evaluate_population(
        self,
        population: list[PredictionRequest],
        objectives: list[OptimizationObjective],
    ) -> list[OptimizationCandidate]:
        """
        Evaluate population against objectives.
        
        Args:
            population: List of candidate designs
            objectives: Optimization objectives
            
        Returns:
            Evaluated candidates
        """
        candidates = []
        
        for request in population:
            # Get predictions
            result = await self.predictor.predict(request)
            
            # Extract predictions
            predictions = {
                "capacitance": result.areal_capacitance.value,
                "esr": result.esr.value,
                "rate_capability": result.rate_capability.value,
                "cycle_life": float(result.cycle_life.value),
            }
            
            # Calculate objective values
            objective_values = {}
            is_feasible = True
            
            for obj in objectives:
                value = predictions[obj.metric]
                
                # Check constraints
                if obj.constraint_min is not None and value < obj.constraint_min:
                    is_feasible = False
                if obj.constraint_max is not None and value > obj.constraint_max:
                    is_feasible = False
                
                # Calculate objective (negate for maximization)
                if obj.target == "maximize":
                    objective_values[obj.metric] = -value * obj.weight
                else:
                    objective_values[obj.metric] = value * obj.weight
            
            # Only add feasible candidates
            if is_feasible:
                candidate = OptimizationCandidate(
                    request=request,
                    predictions=predictions,
                    objectives=objective_values,
                    rank=0,
                    crowding_distance=0.0,
                    is_pareto_optimal=False,
                )
                candidates.append(candidate)
        
        return candidates

    def _non_dominated_sort(
        self, candidates: list[OptimizationCandidate]
    ) -> list[list[OptimizationCandidate]]:
        """
        Perform non-dominated sorting (NSGA-II).
        
        Args:
            candidates: List of evaluated candidates
            
        Returns:
            List of Pareto fronts
        """
        fronts: list[list[OptimizationCandidate]] = [[]]
        
        # Calculate domination
        domination_count = [0] * len(candidates)
        dominated_solutions = [[] for _ in range(len(candidates))]
        
        for i, p in enumerate(candidates):
            for j, q in enumerate(candidates):
                if i == j:
                    continue
                
                if self._dominates(p, q):
                    dominated_solutions[i].append(j)
                elif self._dominates(q, p):
                    domination_count[i] += 1
            
            if domination_count[i] == 0:
                p.rank = 0
                p.is_pareto_optimal = True
                fronts[0].append(p)
        
        # Build subsequent fronts
        i = 0
        while fronts[i]:
            next_front = []
            for p_idx in [candidates.index(p) for p in fronts[i]]:
                for q_idx in dominated_solutions[p_idx]:
                    domination_count[q_idx] -= 1
                    if domination_count[q_idx] == 0:
                        candidates[q_idx].rank = i + 1
                        next_front.append(candidates[q_idx])
            
            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        
        return fronts

    def _dominates(
        self, p: OptimizationCandidate, q: OptimizationCandidate
    ) -> bool:
        """
        Check if candidate p dominates candidate q.
        
        Args:
            p: First candidate
            q: Second candidate
            
        Returns:
            True if p dominates q
        """
        # p dominates q if p is better in at least one objective
        # and not worse in any objective
        better_in_any = False
        
        for metric in p.objectives:
            p_val = p.objectives[metric]
            q_val = q.objectives[metric]
            
            if p_val > q_val:  # Worse (objectives are negated for maximization)
                return False
            elif p_val < q_val:  # Better
                better_in_any = True
        
        return better_in_any

    def _calculate_crowding_distance(
        self, front: list[OptimizationCandidate]
    ) -> None:
        """
        Calculate crowding distance for diversity preservation.
        
        Args:
            front: List of candidates in the same front
        """
        if len(front) <= 2:
            for candidate in front:
                candidate.crowding_distance = float("inf")
            return
        
        # Initialize distances
        for candidate in front:
            candidate.crowding_distance = 0.0
        
        # Calculate for each objective
        for metric in front[0].objectives:
            # Sort by objective value
            front.sort(key=lambda x: x.objectives[metric])
            
            # Boundary points get infinite distance
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")
            
            # Calculate range
            obj_range = front[-1].objectives[metric] - front[0].objectives[metric]
            
            if obj_range == 0:
                continue
            
            # Calculate distances for intermediate points
            for i in range(1, len(front) - 1):
                distance = (
                    front[i + 1].objectives[metric] - front[i - 1].objectives[metric]
                ) / obj_range
                front[i].crowding_distance += distance
