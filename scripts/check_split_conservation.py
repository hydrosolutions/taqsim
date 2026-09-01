import argparse

from _inputs import interval_volume, make_water_system

from taqsim import MonthlyDistribution

p = argparse.ArgumentParser()
p.add_argument("--claim-factor", type=float, required=True)
a = p.parse_args()
b = make_water_system(1, "1 mL")
b.source("river", interval_volume([100.0]))
b.reach(
    "diversion",
    "river",
    "unused",
    rule=MonthlyDistribution({"left": (a.claim_factor / 2,) * 12, "right": (a.claim_factor / 2,) * 12}),
)
model = b.build()
try:
    model.run(bytes([22]) * 16)
except ValueError as error:
    assert "requests" in str(error).lower() and "available" in str(error).lower(), error
else:
    raise AssertionError("incidence accepted a fabricating split")
