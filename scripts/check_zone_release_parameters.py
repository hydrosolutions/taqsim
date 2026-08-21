import argparse

from taqsim import Basin, ZoneRelease, monthly_parameters

p = argparse.ArgumentParser()
p.add_argument("--expected-parameters", type=int, required=True)
a = p.parse_args()
rates = monthly_parameters("seasonal-release-rate", tuple(float(i) for i in range(1, 13)), (0.0, 500.0))
b = Basin(start_date="2020-01-01", timesteps=12)
b.source("river", [1_000_000.0] * 12)
b.reach("reservoir", "river", "downstream", rule=ZoneRelease(100.0, 200.0, 500.0, rates))
m = b.build()
rule = next(item for item in m.document["rules"] if item["compartment"] == "reservoir")
assert len(rule["parameters"]) == a.expected_parameters, rule["parameters"]
