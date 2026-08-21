"""check_resolution_refusals : oversized basin declaration → required refusal."""

import argparse

from taqsim import Basin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basin-total-m3", type=float, required=True)
    parser.add_argument("--resolution", required=True)
    arguments = parser.parse_args()

    basin = Basin(start_date="2020-01-01", timesteps=1, resolution=arguments.resolution)
    basin.source("river", [arguments.basin_total_m3])
    basin.sink("sea")
    basin.reach("channel", "river", "sea")
    try:
        basin.build()
    except ValueError as error:
        message = str(error)
        if "declared total" not in message or "countable ceiling" not in message:
            raise RuntimeError(f"build raised the wrong refusal: {message}") from error
        displayed_total = f"{arguments.basin_total_m3:.17g}"
        if displayed_total not in message:
            raise RuntimeError(f"refusal does not name basin total {displayed_total}: {message}") from error
        return
    raise RuntimeError("build accepted a basin beyond the exactly countable ceiling")


if __name__ == "__main__":
    main()
