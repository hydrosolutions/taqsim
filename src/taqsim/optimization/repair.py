from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from taqsim.system import WaterSystem


def make_repair(system: "WaterSystem") -> Callable[["NDArray[np.float64]"], "NDArray[np.float64]"]:
    """Create a repair function for use with ctrl-freak operators.

    The returned function:
    1. Clips values to bounds
    2. Applies constraint repairs in order
    3. Returns repaired numpy array

    Example usage with ctrl-freak:
        repair = make_repair(system)
        crossover = lambda p1, p2: repair(sbx_crossover(...)(p1, p2))
        mutate = lambda x: repair(polynomial_mutation(...)(x))

    Args:
        system: WaterSystem with param_schema, param_bounds, and constraint_specs

    Returns:
        Repair function that takes and returns numpy arrays
    """
    schema = system.param_schema()
    bounds_dict = system.param_bounds()
    constraint_specs = system.constraint_specs()

    # Build ordered list of keys matching vector indices
    keys = [spec.path for spec in schema]

    # Pre-compute bounds as arrays for fast clipping
    lower = np.array([bounds_dict[k][0] for k in keys])
    upper = np.array([bounds_dict[k][1] for k in keys])

    def repair(x: "NDArray[np.float64]") -> "NDArray[np.float64]":
        # 1. Clip to bounds (fast, vectorized)
        result = np.clip(x, lower, upper)

        # 2. If no constraints, return clipped
        if not constraint_specs:
            return result

        # 3. Convert to dict for constraint operations
        values = {k: float(result[i]) for i, k in enumerate(keys)}

        # 4. Apply each constraint in order
        for spec in constraint_specs:
            # Extract relevant values with local names
            local_values = {p: values[full_path] for p, full_path in spec.param_paths.items()}

            # Apply repair with bounds
            repaired = spec.constraint.repair(local_values, spec.param_bounds)

            # Write back to main dict
            for p, full_path in spec.param_paths.items():
                values[full_path] = repaired[p]

        # 5. Convert back to array
        return np.array([values[k] for k in keys])

    return repair
