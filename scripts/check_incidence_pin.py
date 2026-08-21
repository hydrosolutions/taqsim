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
    quantum = built._document["units"][0]["quantum"]
    if quantum != 1e-3:
        raise RuntimeError(f"decoded model has quantum {quantum!r}, expected 0.001")


if __name__ == "__main__":
    main()
