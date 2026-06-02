"""Multi-objective optimization for MXene supercapacitor design."""

import numpy as np
from typing import Any
from dataclasses import dataclass, field
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
    Multi-objective optimization using NSGA-II.
    
    Finds Pareto-optimal device designs that balance multiple objectives
    (e.g., high capacitance + low ESR + high cycle life).
    """

    # Design-space constants
    MXENE_TYPES = [e.value for e in MXeneType]
    TERMINATIONS = [e.value for e in Termination]
    ELECTROLYTES = [e.value for e in Electrolyte]
    DEPOSITION_METHODS = [e.value for e in DepositionMethod]

    # Continuous parameter bounds (name, min, max, is_optional)
    CONTINUOUS_PARAMS = [
        ("thickness_um",              1.0,   30.0,  False),
        ("electrolyte_concentration", 0.5,   3.0,   True),
        ("annealing_temp_c",          80.0,  200.0, True),
        ("annealing_time_min",        30.0,  120.0, True),
        ("interlayer_spacing_nm",     0.9,   1.8,   True),
        ("specific_surface_area_m2g", 50.0,  150.0, True),
        ("pore_volume_cm3g",          0.05,  0.25,  True),
    ]

    def __init__(self, predictor: Any) -> None:
        self.predictor = predictor
        self._rng = np.random.default_rng(42)

    # ── public API ────────────────────────────────────────────────────────────

    async def optimize(
        self,
        objectives: list[OptimizationObjective],
        constraints: dict[str, Any] | None = None,
        population_size: int = 100,
        generations: int = 50,
    ) -> list[OptimizationCandidate]:
        """
        Run NSGA-II multi-objective optimization.

        Args:
            objectives: Optimization objectives
            constraints: Design-space constraints
            population_size: Individuals per generation
            generations: Number of evolution iterations

        Returns:
            Pareto-optimal candidates sorted by crowding distance
        """
        # 1. Generate & evaluate initial population
        population = self._generate_initial_population(population_size, constraints)
        evaluated = await self._evaluate_population(population, objectives)

        # 2. Evolution loop ─ real NSGA-II
        for _gen in range(generations):
            # a) Create offspring via selection + crossover + mutation
            offspring_requests = self._create_offspring(
                evaluated, population_size, constraints,
            )

            # b) Evaluate offspring
            offspring_evaluated = await self._evaluate_population(
                offspring_requests, objectives,
            )

            # c) Merge parents + offspring
            combined = evaluated + offspring_evaluated

            # d) Non-dominated sort on combined pool
            fronts = self._non_dominated_sort(combined)

            # e) Select next generation (fill up to population_size)
            next_gen: list[OptimizationCandidate] = []
            for front in fronts:
                if len(next_gen) + len(front) <= population_size:
                    next_gen.extend(front)
                else:
                    # Need only some from this front → pick by crowding distance
                    self._calculate_crowding_distance(front)
                    front.sort(key=lambda c: c.crowding_distance, reverse=True)
                    remaining = population_size - len(next_gen)
                    next_gen.extend(front[:remaining])
                    break

            evaluated = next_gen

        # 3. Final sort & return Pareto front
        fronts = self._non_dominated_sort(evaluated)
        pareto_optimal = fronts[0] if fronts else []

        if pareto_optimal:
            self._calculate_crowding_distance(pareto_optimal)
            pareto_optimal.sort(key=lambda c: c.crowding_distance, reverse=True)

        return pareto_optimal

    # ── genetic operators ─────────────────────────────────────────────────────

    def _create_offspring(
        self,
        parents: list[OptimizationCandidate],
        n_offspring: int,
        constraints: dict[str, Any] | None,
    ) -> list[PredictionRequest]:
        """Generate offspring via tournament selection, crossover, mutation."""
        offspring: list[PredictionRequest] = []

        while len(offspring) < n_offspring:
            p1 = self._tournament_select(parents)
            p2 = self._tournament_select(parents)
            child = self._crossover(p1.request, p2.request)
            child = self._mutate(child, constraints)
            offspring.append(child)

        return offspring[:n_offspring]

    def _tournament_select(
        self, pop: list[OptimizationCandidate], k: int = 3,
    ) -> OptimizationCandidate:
        """Binary tournament: pick *k* random, return best by (rank, -crowding)."""
        indices = self._rng.choice(len(pop), size=min(k, len(pop)), replace=False)
        contenders = [pop[i] for i in indices]
        return min(contenders, key=lambda c: (c.rank, -c.crowding_distance))

    def _crossover(
        self, p1: PredictionRequest, p2: PredictionRequest,
    ) -> PredictionRequest:
        """Simulated binary crossover (SBX) for continuous + uniform swap for categorical."""
        d = p1.model_dump()
        d2 = p2.model_dump()

        # Categorical: 50 / 50 swap
        for key in ("mxene_type", "terminations", "electrolyte", "deposition_method"):
            if self._rng.random() < 0.5:
                d[key] = d2[key]

        # Continuous: SBX (eta=2)
        eta = 2.0
        for name, lo, hi, _ in self.CONTINUOUS_PARAMS:
            v1 = d.get(name)
            v2 = d2.get(name)
            if v1 is None or v2 is None:
                # If one parent has it and the other doesn't, 50/50
                d[name] = v1 if self._rng.random() < 0.5 else v2
                continue
            u = self._rng.random()
            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (eta + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
            child_val = 0.5 * ((1 + beta) * v1 + (1 - beta) * v2)
            d[name] = float(np.clip(child_val, lo, hi))

        return PredictionRequest(**d)

    def _mutate(
        self,
        individual: PredictionRequest,
        constraints: dict[str, Any] | None,
        mutation_rate: float = 0.15,
    ) -> PredictionRequest:
        """Polynomial mutation for continuous, random swap for categorical."""
        d = individual.model_dump()

        # Categorical mutation
        if self._rng.random() < mutation_rate:
            d["mxene_type"] = self._rng.choice(self.MXENE_TYPES)
        if self._rng.random() < mutation_rate:
            d["terminations"] = self._rng.choice(self.TERMINATIONS)
        if self._rng.random() < mutation_rate:
            d["electrolyte"] = self._rng.choice(self.ELECTROLYTES)
        if self._rng.random() < mutation_rate:
            d["deposition_method"] = self._rng.choice(self.DEPOSITION_METHODS)

        # Continuous: polynomial mutation (eta_m = 20)
        eta_m = 20.0
        for name, lo, hi, is_opt in self.CONTINUOUS_PARAMS:
            if d.get(name) is None:
                continue
            if self._rng.random() >= mutation_rate:
                continue
            val = d[name]
            delta = (val - lo) / (hi - lo) if hi != lo else 0.5
            u = self._rng.random()
            if u < 0.5:
                deltaq = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
            else:
                deltaq = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
            val_new = val + deltaq * (hi - lo)
            d[name] = float(np.clip(val_new, lo, hi))

        # Apply external constraints if provided
        if constraints:
            lo_t = constraints.get("thickness_min", 1.0)
            hi_t = constraints.get("thickness_max", 30.0)
            d["thickness_um"] = float(np.clip(d["thickness_um"], lo_t, hi_t))

        return PredictionRequest(**d)

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
        rng = self._rng
        
        # Apply constraints
        thickness_min = constraints.get("thickness_min", 1.0) if constraints else 1.0
        thickness_max = constraints.get("thickness_max", 30.0) if constraints else 30.0
        
        for i in range(size):
            # Sample categorical variables
            mxene_type = rng.choice(self.MXENE_TYPES)
            termination = rng.choice(self.TERMINATIONS)
            electrolyte = rng.choice(self.ELECTROLYTES)
            deposition_method = rng.choice(self.DEPOSITION_METHODS)
            
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
        
        # Build index map for O(1) lookup (fixes O(n³) from list.index())
        idx_map = {id(c): i for i, c in enumerate(candidates)}

        # Build subsequent fronts
        fi = 0
        while fronts[fi]:
            next_front = []
            for p in fronts[fi]:
                p_idx = idx_map[id(p)]
                for q_idx in dominated_solutions[p_idx]:
                    domination_count[q_idx] -= 1
                    if domination_count[q_idx] == 0:
                        candidates[q_idx].rank = fi + 1
                        next_front.append(candidates[q_idx])
            fi += 1
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
