import argparse

from _inputs import interval_volume, make_water_system

from taqsim import MonthlyDistribution, monthly_parameters

p = argparse.ArgumentParser()
p.add_argument("--expected-parameters", type=int, required=True)
a = p.parse_args()
rates = monthly_parameters("seasonal-release-rate", tuple(float(i) for i in range(1, 13)), (0.0, 500.0))
b = make_water_system(12, "1 mL")
b.source("river", interval_volume([1_000_000.0] * 12))
b.reach("reservoir", "river", "downstream", rule=MonthlyDistribution({"downstream": rates}))
m = b.build()
rule = next(item for item in m.document["rules"] if item["compartment"] == "reservoir")
assert len(rule["parameters"]) == a.expected_parameters, rule["parameters"]
