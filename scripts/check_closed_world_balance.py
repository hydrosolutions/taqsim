import argparse
import math

from taqsim import Basin

p = argparse.ArgumentParser()
p.add_argument("--atol", type=float, required=True)
a = p.parse_args()
flows = [15.0, 25.0, 7.0, 30.0]
b = Basin(start_date="2020-01-01", timesteps=len(flows), resolution="1 mL")
b.source("river", flows)
b.reach("canal", "river", "farm", capacity=10.0, overflow_destination="floodplain")
run = b.build().run(bytes([23]) * 16)
farm = sum(value or 0.0 for value in run.arrivals("farm").values)
flood = sum(value or 0.0 for value in run.arrivals("floodplain").values)
assert math.isclose(farm + flood, sum(flows), rel_tol=0.0, abs_tol=a.atol), (farm, flood, sum(flows))
assert flood > 0.0
