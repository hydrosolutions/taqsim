"""Witness that a taqsim parameter sweep compiles once and submits only run data."""

from __future__ import annotations

import argparse

import incidence
from _inputs import interval_volume, make_water_system

from taqsim import Parameter, VolumetricRate, WaterVolume, ZoneRelease


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, required=True)
    args = parser.parse_args()
    if args.trials < 50:
        raise ValueError("the held-model witness requires at least 50 trials")

    compile_calls: list[dict[str, object]] = []
    run_calls: list[tuple[bytes, list[dict[str, object]]]] = []
    original_compile = incidence.compile_model

    class HeldModel:
        def __init__(self, compiled: incidence.CompiledModel) -> None:
            self._compiled = compiled
            self.model_digest = compiled.model_digest

        def quantum(self, substance: str) -> float:
            return self._compiled.quantum(substance)

        def run(self, run_id: bytes, *, substitutions: list[dict[str, object]]) -> incidence.CompletedRun:
            run_calls.append((run_id, substitutions))
            return self._compiled.run(run_id, substitutions=substitutions)

    def observed_compile(document: dict[str, object]) -> HeldModel:
        compile_calls.append(document)
        return HeldModel(original_compile(document))

    incidence.compile_model = observed_compile
    try:
        water_system = make_water_system(1, "1 mL")
        water_system.source("river", interval_volume([100.0]))
        water_system.reach(
            "reservoir",
            "river",
            "downstream",
            rule=ZoneRelease(
                WaterVolume(0.0, "m3"),
                WaterVolume(0.0, "m3"),
                WaterVolume(1_000.0, "m3"),
                VolumetricRate(Parameter("release-rate", 1.0, (0.0, 2.0)), "m3/s"),
            ),
        )
        model = water_system.build()
        runs = model.sweep(
            "reservoir.release-rate",
            (VolumetricRate(2.0 * trial / (args.trials - 1), "m3/s") for trial in range(args.trials)),
        )
    finally:
        incidence.compile_model = original_compile

    assert len(compile_calls) == 1, f"model compiled {len(compile_calls)} times"
    assert len(runs) == args.trials
    assert len(run_calls) == args.trials
    for run_id, substitutions in run_calls:
        assert isinstance(run_id, bytes) and len(run_id) == 16
        assert substitutions == [
            {
                "compartment": "reservoir",
                "substance": "water",
                "parameter": "release-rate",
                "value": substitutions[0]["value"],
            }
        ]


if __name__ == "__main__":
    main()
