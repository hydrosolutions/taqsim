import argparse
import math

from _inputs import interval_volume, make_water_system

from taqsim import WaterVolume

p = argparse.ArgumentParser()
p.add_argument("--atol", type=float, required=True)
a = p.parse_args()
flows = [15.0, 25.0, 7.0, 30.0]
b = make_water_system(len(flows), "1 mL")
b.source("river", interval_volume(flows))
b.reach("canal", "river", "farm", capacity=WaterVolume(10.0, "m3"), overflow_destination="floodplain")
run = b.build().run(bytes([23]) * 16)
farm = sum(value or 0.0 for value in run.arrivals("farm").values)
flood = sum(value or 0.0 for value in run.arrivals("floodplain").values)
assert math.isclose(farm + flood, sum(flows), rel_tol=0.0, abs_tol=a.atol), (farm, flood, sum(flows))
assert flood > 0.0
