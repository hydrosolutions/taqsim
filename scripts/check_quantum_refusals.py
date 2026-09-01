"""check_quantum_refusals : oversized water_system declaration → required refusal."""

import argparse

from _inputs import interval_volume, make_water_system


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--water-system-total-m3", type=float, required=True)
    parser.add_argument("--quantum", required=True)
    arguments = parser.parse_args()

    water_system = make_water_system(1, arguments.quantum)
    water_system.source("river", interval_volume([arguments.water_system_total_m3]))
    water_system.sink("sea")
    water_system.reach("channel", "river", "sea")
    try:
        water_system.build()
    except ValueError as error:
        message = str(error)
        if "countable ceiling" not in message:
            raise RuntimeError(f"build raised the wrong refusal: {message}") from error
        displayed_total = f"{arguments.water_system_total_m3:.17g}"
        if displayed_total not in message:
            raise RuntimeError(f"refusal does not name water system total {displayed_total}: {message}") from error
        return
    raise RuntimeError("build accepted a water system beyond the exactly countable ceiling")


if __name__ == "__main__":
    main()
