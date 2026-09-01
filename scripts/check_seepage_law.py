"""check_seepage_law : DrawCount × Resolution × ToleranceUnits → None.

Sample physically admissible canals and fail if their recorded seepage is farther
from the hydraulic law than the requested number of resolution units.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass

from taqsim import Basin, CanalLosses, Resolution

_SEED = 11
_SECONDS_PER_TIMESTEP = 86_400.0
_COEFFICIENT_RANGE = (1e-4, 1e-2)
_LENGTH_KM_RANGE = (0.5, 15.0)
_FLOW_RANGE = (100.0, 10_000.0)


@dataclass(frozen=True)
class Draw:
    """One canal whose uncapped hydraulic loss does not exceed its flow."""

    coefficient: float
    length_km: float
    flow_m3: float
    expected_seepage_m3: float


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def _draw_canals(count: int, resolution: Resolution) -> tuple[Draw, ...]:
    generator = random.Random(_SEED)
    draws: list[Draw] = []
    while len(draws) < count:
        coefficient = generator.uniform(*_COEFFICIENT_RANGE)
        length_km = generator.uniform(*_LENGTH_KM_RANGE)
        sampled_flow = generator.uniform(*_FLOW_RANGE)
        flow_m3 = math.floor(sampled_flow / resolution.quantum_m3) * resolution.quantum_m3
        flow_m3s = flow_m3 / _SECONDS_PER_TIMESTEP
        expected = coefficient * math.sqrt(flow_m3s) * length_km * _SECONDS_PER_TIMESTEP
        # CanalLosses deliberately caps seepage at the available stock. The gate
        # samples the uncapped part of the law so it tests the formula itself.
        if expected <= flow_m3:
            draws.append(Draw(coefficient, length_km, flow_m3, expected))
    return tuple(draws)


def _recorded_seepage(draw: Draw, resolution: Resolution, run_number: int) -> float:
    basin = Basin(start_date="2020-01-01", timesteps=1, resolution=resolution)
    basin.source("river", [draw.flow_m3])
    basin.reach("canal", "river", "farm", rule=CanalLosses(draw.coefficient, draw.length_km))
    run = basin.build().run(run_number.to_bytes(16, "big"))
    recorded = run.arrivals("seepage").values[0]
    if recorded is None:
        raise RuntimeError(f"draw {run_number} returned absent seepage")
    return recorded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=_positive_int, required=True)
    parser.add_argument("--resolution", choices=tuple(item.value for item in Resolution), required=True)
    parser.add_argument("--tolerance-units", type=_non_negative_float, required=True)
    arguments = parser.parse_args()

    resolution = Resolution(arguments.resolution)
    tolerance_m3 = arguments.tolerance_units * resolution.quantum_m3
    for run_number, draw in enumerate(_draw_canals(arguments.draws, resolution), start=1):
        recorded = _recorded_seepage(draw, resolution, run_number)
        error = abs(recorded - draw.expected_seepage_m3)
        if not math.isfinite(recorded) or error >= tolerance_m3:
            raise RuntimeError(
                f"draw {run_number} violates the seepage law: recorded={recorded!r} m3, "
                f"expected={draw.expected_seepage_m3!r} m3, error={error!r} m3, "
                f"tolerance={tolerance_m3!r} m3; coefficient={draw.coefficient!r}, "
                f"length_km={draw.length_km!r}, flow_m3={draw.flow_m3!r}"
            )


if __name__ == "__main__":
    main()
