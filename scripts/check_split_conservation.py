import argparse

from taqsim import Basin, MonthlyDistribution

p = argparse.ArgumentParser()
p.add_argument("--claim-factor", type=float, required=True)
a = p.parse_args()
b = Basin(start_date="2020-01-01", timesteps=1, resolution="1 mL")
b.source("river", [100.0])
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
