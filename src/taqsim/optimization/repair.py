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
    2. Applies constraint repairs (per-timestep for time-varying params)
    3. Returns repaired numpy array
    """
    schema = system.param_schema()
    bounds_dict = system.param_bounds()
    constraint_specs = system.constraint_specs()

    # Build ordered list of keys matching vector indices
    keys = [spec.path for spec in schema]

    # Pre-compute bounds as arrays for fast clipping
    lower = np.array([bounds_dict[k][0] for k in keys])
    upper = np.array([bounds_dict[k][1] for k in keys])

    # Build index lookup: path -> vector index
    path_to_idx = {path: i for i, path in enumerate(keys)}

    # For time-varying constraints, determine the number of timesteps
    # by looking at paths like "node.strategy.param[0]", "node.strategy.param[1]", etc.
    def get_timestep_count(spec):
        """Get number of timesteps for a time-varying constraint."""
        if not spec.time_varying_params:
            return 0
        # Find a time-varying param and count its indices
        for local_name in spec.time_varying_params:
            base_path = spec.param_paths[local_name]
            count = 0
            while f"{base_path}[{count}]" in path_to_idx:
                count += 1
            if count > 0:
                return count
        return 0

    def repair(x: "NDArray[np.float64]") -> "NDArray[np.float64]":
        # 1. Clip to bounds (fast, vectorized)
        result = np.clip(x, lower, upper)

        if not constraint_specs:
            return result

        # 2. Apply each constraint
        for spec in constraint_specs:
            if spec.time_varying_params:
                # Time-varying constraint: apply per-timestep
                n_timesteps = get_timestep_count(spec)
                for t in range(n_timesteps):
                    # Extract scalar values for this timestep
                    local_values = {}
                    for p, full_path in spec.param_paths.items():
                        if p in spec.time_varying_params:
                            # Time-varying: get indexed value
                            idx = path_to_idx[f"{full_path}[{t}]"]
                            local_values[p] = float(result[idx])
                        else:
                            # Constant: get scalar value
                            idx = path_to_idx[full_path]
                            local_values[p] = float(result[idx])

                    # Apply repair
                    repaired = spec.constraint.repair(local_values, spec.param_bounds)

                    # Write back ONLY time-varying params
                    for p, full_path in spec.param_paths.items():
                        if p in spec.time_varying_params:
                            idx = path_to_idx[f"{full_path}[{t}]"]
                            result[idx] = repaired[p]
            else:
                # Constant-only constraint: apply once (original behavior)
                local_values = {p: float(result[path_to_idx[full_path]]) for p, full_path in spec.param_paths.items()}
                repaired = spec.constraint.repair(local_values, spec.param_bounds)
                for p, full_path in spec.param_paths.items():
                    result[path_to_idx[full_path]] = repaired[p]

        return result

    return repair
