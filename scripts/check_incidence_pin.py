"""check_incidence_pin : required schema capability → process success or refusal."""

import argparse

from _inputs import interval_volume, make_water_system


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-quantum-field", action="store_true", required=True)
    parser.parse_args()

    water_system = make_water_system(1, "1 L")
    water_system.source("river", interval_volume([1.0]))
    water_system.sink("sea")
    water_system.reach("channel", "river", "sea")
    built = water_system.build()
    run = built.run(bytes([91]) * 16)
    if built.quantum.quantum_m3 != 1e-3 or run.quantum is not built.quantum:
        raise RuntimeError(f"model/run quantum mismatch: model={built.quantum!r}, run={run.quantum!r}")


if __name__ == "__main__":
    main()
