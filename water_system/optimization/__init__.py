"""
optimization

This subpackage provides optimization algorithms and visualization tools for multi-objective
and single-objective optimization of water system simulations.

Modules and Classes:
    - deap_optimization: Contains DEAP-based genetic algorithm optimizers for single and multi-objective problems.
        * DeapSingleObjectiveOptimizer
        * DeapTwoObjectiveOptimizer
        * DeapThreeObjectiveOptimizer
        * DeapFourObjectiveOptimizer

    - pymoo_optimization: Provides wrappers for Pymoo-based optimization algorithms.
        * PymooSingleObjectiveOptimizer
        * PymooMultiObjectiveOptimizer

    - pareto_dashboard: Interactive dashboards for visualizing Pareto fronts in 3D and 4D.
        * ParetoFrontDashboard3D
        * ParetoFrontDashboard4D

Usage:
    from water_system.optimization import (
        DeapSingleObjectiveOptimizer, DeapTwoObjectiveOptimizer,
        DeapThreeObjectiveOptimizer, DeapFourObjectiveOptimizer,
        PymooSingleObjectiveOptimizer, PymooMultiObjectiveOptimizer,
        ParetoFrontDashboard3D, ParetoFrontDashboard4D
    )

This allows users to easily access all major optimization and visualization tools for
water system model calibration and analysis.
"""

from .optimizer import DeapOptimizer
from .pareto_dashboard import (
    ParetoFrontDashboard3D, ParetoFrontDashboard4D
)

__all__ = [
    'ParetoFrontDashboard3D', 'ParetoFrontDashboard4D', 'DeapOptimizer'
]