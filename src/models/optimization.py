"""
Multi-objective optimization for cost-performance tradeoff analysis.

This module provides Pareto-optimal configuration discovery using
multi-objective optimization techniques.

Course: DATA 230 (Data Visualization) at SJSU
"""

from typing import Optional, Callable, Any
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

from src.core.config import ModelConfig, get_model_config
from src.core.exceptions import PredictionError


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationObjective:
    """Definition of an optimization objective."""
    
    name: str
    direction: str  # "minimize" or "maximize"
    weight: float = 1.0
    
    def __post_init__(self) -> None:
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError(f"direction must be 'minimize' or 'maximize', got {self.direction}")
        if self.weight <= 0:
            raise ValueError(f"weight must be positive, got {self.weight}")


@dataclass(frozen=True)
class ParetoSolution:
    """A solution on the Pareto frontier."""
    
    configuration: dict[str, Any]
    objective_values: dict[str, float]
    rank: int
    crowding_distance: float
    
    @property
    def is_pareto_optimal(self) -> bool:
        """Check if this solution is Pareto optimal (rank 0)."""
        return self.rank == 0


class ParetoFrontierExtractor:
    """
    Extract Pareto-optimal solutions from a set of configurations.
    
    Uses non-dominated sorting to identify solutions that represent
    optimal tradeoffs between multiple objectives.
    
    Example:
        >>> extractor = ParetoFrontierExtractor()
        >>> pareto_solutions = extractor.extract(
        ...     configurations=configs,
        ...     objectives=[
        ...         OptimizationObjective("performance", "maximize"),
        ...         OptimizationObjective("cost", "minimize")
        ...     ]
        ... )
    """
    
    def __init__(self) -> None:
        """Initialize Pareto frontier extractor."""
        logger.info("Initialized ParetoFrontierExtractor")
    
    def extract(
        self,
        configurations: list[dict[str, Any]],
        objectives: list[OptimizationObjective]
    ) -> list[ParetoSolution]:
        """
        Extract Pareto-optimal solutions.
        
        Args:
            configurations: List of configuration dictionaries.
            objectives: List of optimization objectives.
            
        Returns:
            List of Pareto solutions sorted by rank and crowding distance.
        """
        if not configurations:
            return []
        
        n = len(configurations)
        
        # Extract objective values
        objective_matrix = np.zeros((n, len(objectives)))
        for i, config in enumerate(configurations):
            for j, obj in enumerate(objectives):
                value = config.get(obj.name, 0)
                # Negate for minimization to make all objectives "maximize"
                if obj.direction == "minimize":
                    value = -value
                objective_matrix[i, j] = value * obj.weight
        
        # Non-dominated sorting
        ranks = self._non_dominated_sort(objective_matrix)
        
        # Calculate crowding distance
        crowding = self._calculate_crowding_distance(objective_matrix, ranks)
        
        # Build solutions
        solutions = []
        for i, config in enumerate(configurations):
            obj_values = {obj.name: config.get(obj.name, 0) for obj in objectives}
            
            solution = ParetoSolution(
                configuration=config,
                objective_values=obj_values,
                rank=ranks[i],
                crowding_distance=crowding[i]
            )
            solutions.append(solution)
        
        # Sort by rank, then by crowding distance (descending)
        solutions.sort(key=lambda s: (s.rank, -s.crowding_distance))
        
        logger.info(
            "Pareto frontier extracted",
            extra={
                "total_solutions": n,
                "pareto_optimal": sum(1 for s in solutions if s.is_pareto_optimal)
            }
        )
        
        return solutions
    
    def _non_dominated_sort(self, objectives: np.ndarray) -> np.ndarray:
        """
        Perform non-dominated sorting.
        
        Args:
            objectives: Matrix of objective values (n_solutions x n_objectives).
            
        Returns:
            Array of ranks for each solution.
        """
        n = len(objectives)
        ranks = np.zeros(n, dtype=int)
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions = [[] for _ in range(n)]
        
        # Calculate domination relationships
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(objectives[i], objectives[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(objectives[j], objectives[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1
        
        # Assign ranks
        current_rank = 0
        current_front = [i for i in range(n) if domination_count[i] == 0]
        
        while current_front:
            for i in current_front:
                ranks[i] = current_rank
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
            
            current_rank += 1
            current_front = [i for i in range(n) 
                          if domination_count[i] == 0 and ranks[i] == 0 and i not in current_front]
            
            # Rebuild front for next iteration
            current_front = [i for i in range(n) if domination_count[i] == 0 and ranks[i] == current_rank - 1]
            current_front = []
            for i in range(n):
                if domination_count[i] == 0 and ranks[i] == 0:
                    current_front.append(i)
                    ranks[i] = current_rank
            
            if not any(domination_count[i] == 0 and ranks[i] == 0 for i in range(n)):
                break
        
        return ranks
    
    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Check if solution a dominates solution b."""
        return np.all(a >= b) and np.any(a > b)
    
    def _calculate_crowding_distance(
        self,
        objectives: np.ndarray,
        ranks: np.ndarray
    ) -> np.ndarray:
        """
        Calculate crowding distance for diversity preservation.
        
        Args:
            objectives: Matrix of objective values.
            ranks: Array of ranks.
            
        Returns:
            Array of crowding distances.
        """
        n = len(objectives)
        crowding = np.zeros(n)
        
        for rank in range(max(ranks) + 1):
            front_indices = np.where(ranks == rank)[0]
            if len(front_indices) <= 2:
                crowding[front_indices] = np.inf
                continue
            
            for m in range(objectives.shape[1]):
                sorted_indices = front_indices[np.argsort(objectives[front_indices, m])]
                
                # Boundary points get infinite distance
                crowding[sorted_indices[0]] = np.inf
                crowding[sorted_indices[-1]] = np.inf
                
                # Calculate distance for interior points
                obj_range = objectives[sorted_indices[-1], m] - objectives[sorted_indices[0], m]
                if obj_range > 0:
                    for i in range(1, len(sorted_indices) - 1):
                        crowding[sorted_indices[i]] += (
                            objectives[sorted_indices[i + 1], m] -
                            objectives[sorted_indices[i - 1], m]
                        ) / obj_range
        
        return crowding


class CostPerformanceTradeoffAnalyzer:
    """
    Analyze cost-performance tradeoffs using multi-objective optimization.
    
    Identifies optimal configurations that balance cost, performance,
    and risk objectives.
    
    Example:
        >>> analyzer = CostPerformanceTradeoffAnalyzer()
        >>> tradeoffs = analyzer.analyze(
        ...     data=agent_data,
        ...     performance_col="business_value_score",
        ...     cost_col="cost_per_task_cents",
        ...     risk_col="operational_risk_index"
        ... )
    """
    
    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        """
        Initialize tradeoff analyzer.
        
        Args:
            config: Optional model configuration.
        """
        self._config = config or get_model_config()
        self._pareto_extractor = ParetoFrontierExtractor()
        
        logger.info("Initialized CostPerformanceTradeoffAnalyzer")
    
    def analyze(
        self,
        data: pd.DataFrame,
        performance_col: str = "business_value_score",
        cost_col: str = "cost_per_task_cents",
        risk_col: Optional[str] = "operational_risk_index",
        group_cols: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """
        Analyze cost-performance tradeoffs.
        
        Args:
            data: DataFrame with agent configurations.
            performance_col: Column name for performance metric.
            cost_col: Column name for cost metric.
            risk_col: Optional column name for risk metric.
            group_cols: Optional columns to include in configurations.
            
        Returns:
            Dictionary with analysis results.
        """
        # Build configurations
        configs = []
        for _, row in data.iterrows():
            config = {
                "performance": row[performance_col],
                "cost": row[cost_col],
            }
            
            if risk_col and risk_col in data.columns:
                config["risk"] = row[risk_col]
            
            if group_cols:
                for col in group_cols:
                    if col in data.columns:
                        config[col] = row[col]
            
            configs.append(config)
        
        # Define objectives
        objectives = [
            OptimizationObjective("performance", "maximize"),
            OptimizationObjective("cost", "minimize"),
        ]
        
        if risk_col and risk_col in data.columns:
            objectives.append(OptimizationObjective("risk", "minimize"))
        
        # Extract Pareto frontier
        pareto_solutions = self._pareto_extractor.extract(configs, objectives)
        
        # Calculate sweet spots
        sweet_spots = self._identify_sweet_spots(pareto_solutions)
        
        # Calculate tradeoff curves
        tradeoff_curves = self._calculate_tradeoff_curves(pareto_solutions)
        
        return {
            "pareto_solutions": pareto_solutions,
            "pareto_optimal_count": sum(1 for s in pareto_solutions if s.is_pareto_optimal),
            "sweet_spots": sweet_spots,
            "tradeoff_curves": tradeoff_curves,
            "total_configurations": len(configs),
        }
    
    def _identify_sweet_spots(
        self,
        solutions: list[ParetoSolution]
    ) -> list[ParetoSolution]:
        """
        Identify sweet spot configurations with best balance.
        
        Sweet spots are Pareto-optimal solutions with high crowding distance
        (representing unique tradeoff regions).
        """
        pareto_optimal = [s for s in solutions if s.is_pareto_optimal]
        
        if len(pareto_optimal) <= 3:
            return pareto_optimal
        
        # Sort by crowding distance and take top solutions
        sorted_solutions = sorted(
            pareto_optimal,
            key=lambda s: s.crowding_distance,
            reverse=True
        )
        
        # Return solutions with finite crowding distance (not boundary points)
        sweet_spots = [
            s for s in sorted_solutions
            if s.crowding_distance < np.inf
        ][:5]
        
        # Include boundary points if we have few sweet spots
        if len(sweet_spots) < 3:
            sweet_spots = sorted_solutions[:5]
        
        return sweet_spots
    
    def _calculate_tradeoff_curves(
        self,
        solutions: list[ParetoSolution]
    ) -> dict[str, list[tuple[float, float]]]:
        """Calculate pairwise tradeoff curves."""
        pareto_optimal = [s for s in solutions if s.is_pareto_optimal]
        
        curves = {}
        
        if not pareto_optimal:
            return curves
        
        # Performance vs Cost curve
        perf_cost = [
            (s.objective_values.get("performance", 0), s.objective_values.get("cost", 0))
            for s in pareto_optimal
        ]
        curves["performance_vs_cost"] = sorted(perf_cost, key=lambda x: x[0])
        
        # Performance vs Risk curve (if available)
        if "risk" in pareto_optimal[0].objective_values:
            perf_risk = [
                (s.objective_values.get("performance", 0), s.objective_values.get("risk", 0))
                for s in pareto_optimal
            ]
            curves["performance_vs_risk"] = sorted(perf_risk, key=lambda x: x[0])
        
        return curves


class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer using Optuna for hyperparameter search.
    
    Finds optimal configurations across multiple competing objectives
    using advanced optimization algorithms.
    
    Example:
        >>> optimizer = MultiObjectiveOptimizer()
        >>> best_configs = optimizer.optimize(
        ...     objective_fn=evaluate_config,
        ...     param_space=param_definitions,
        ...     n_trials=100
        ... )
    """
    
    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        """
        Initialize multi-objective optimizer.
        
        Args:
            config: Optional model configuration.
        """
        self._config = config or get_model_config()
        
        logger.info("Initialized MultiObjectiveOptimizer")
    
    def optimize(
        self,
        objective_fn: Callable[[dict[str, Any]], tuple[float, ...]],
        param_space: dict[str, Any],
        n_trials: Optional[int] = None,
        n_objectives: int = 2
    ) -> list[dict[str, Any]]:
        """
        Run multi-objective optimization.
        
        Args:
            objective_fn: Function that takes params and returns objective values.
            param_space: Dictionary defining parameter search space.
            n_trials: Number of optimization trials.
            n_objectives: Number of objectives to optimize.
            
        Returns:
            List of Pareto-optimal configurations.
        """
        try:
            import optuna
            from optuna.samplers import NSGAIISampler
        except ImportError:
            logger.warning("optuna not available, using random search")
            return self._random_search(objective_fn, param_space, n_trials or 50)
        
        n_trials = n_trials or self._config.optuna_n_trials
        
        # Create multi-objective study
        study = optuna.create_study(
            directions=["maximize"] * n_objectives,
            sampler=NSGAIISampler()
        )
        
        def optuna_objective(trial: optuna.Trial) -> tuple[float, ...]:
            params = self._sample_params(trial, param_space)
            return objective_fn(params)
        
        study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=False)
        
        # Extract Pareto-optimal trials
        pareto_trials = study.best_trials
        
        results = []
        for trial in pareto_trials:
            results.append({
                "params": trial.params,
                "values": trial.values,
            })
        
        logger.info(
            "Optimization completed",
            extra={
                "n_trials": n_trials,
                "pareto_optimal": len(results)
            }
        )
        
        return results
    
    def _sample_params(
        self,
        trial: Any,
        param_space: dict[str, Any]
    ) -> dict[str, Any]:
        """Sample parameters from search space."""
        params = {}
        
        for name, spec in param_space.items():
            if spec["type"] == "float":
                params[name] = trial.suggest_float(
                    name,
                    spec.get("low", 0),
                    spec.get("high", 1)
                )
            elif spec["type"] == "int":
                params[name] = trial.suggest_int(
                    name,
                    spec.get("low", 0),
                    spec.get("high", 100)
                )
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(
                    name,
                    spec.get("choices", [])
                )
        
        return params
    
    def _random_search(
        self,
        objective_fn: Callable,
        param_space: dict[str, Any],
        n_trials: int
    ) -> list[dict[str, Any]]:
        """Fallback random search when Optuna is not available."""
        results = []
        
        for _ in range(n_trials):
            params = {}
            for name, spec in param_space.items():
                if spec["type"] == "float":
                    params[name] = np.random.uniform(
                        spec.get("low", 0),
                        spec.get("high", 1)
                    )
                elif spec["type"] == "int":
                    params[name] = np.random.randint(
                        spec.get("low", 0),
                        spec.get("high", 100)
                    )
                elif spec["type"] == "categorical":
                    choices = spec.get("choices", [])
                    if choices:
                        params[name] = np.random.choice(choices)
            
            values = objective_fn(params)
            results.append({"params": params, "values": values})
        
        return results
