"""check_incidence_pin : required schema capability → process success or refusal."""

import argparse

from taqsim import Basin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-quantum-field", action="store_true", required=True)
    parser.parse_args()

    basin = Basin(start_date="2020-01-01", timesteps=1, resolution="1 L")
    basin.source("river", [1.0])
    basin.sink("sea")
    basin.reach("channel", "river", "sea")
    built = basin.build()
    run = built.run(bytes([91]) * 16)
    if built.resolution.quantum_m3 != 1e-3 or run.resolution is not built.resolution:
        raise RuntimeError(f"model/run resolution mismatch: model={built.resolution!r}, run={run.resolution!r}")


if __name__ == "__main__":
    main()
