import argparse

from _inputs import interval_volume, make_water_system

from taqsim import Parameter, VolumetricRate, WaterVolume, ZoneRelease

p = argparse.ArgumentParser()
p.add_argument("--expected-parameters", type=int, required=True)
a = p.parse_args()
rates = tuple(
    VolumetricRate(Parameter(f"seasonal-release-rate-{month:02d}", float(month), (0.0, 500.0)), "m3/s")
    for month in range(1, 13)
)
b = make_water_system(12, "1 mL")
b.source("river", interval_volume([1_000_000.0] * 12))
b.reach(
    "reservoir",
    "river",
    "downstream",
    rule=ZoneRelease(
        WaterVolume(100.0, "m3"),
        WaterVolume(200.0, "m3"),
        WaterVolume(500.0, "m3"),
        rates,
    ),
)
m = b.build()
rule = next(item for item in m.document["rules"] if item["compartment"] == "reservoir")
assert len(rule["parameters"]) == a.expected_parameters, rule["parameters"]
