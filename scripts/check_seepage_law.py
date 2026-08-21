import argparse
import math

from taqsim import Basin, CanalLosses

p = argparse.ArgumentParser()
p.add_argument("--rtol", type=float, required=True)
a = p.parse_args()
alpha, length, q, seconds = 0.05, 2.5, 40000.0, 86400.0
b = Basin(start_date="2020-01-01", timesteps=1, resolution="1 mL")
b.source("river", [q])
b.reach("canal", "river", "farm", rule=CanalLosses(alpha, length))
run = b.build().run(bytes([21]) * 16)
actual = run.arrivals("seepage").values[0]
expected = alpha * math.sqrt(q / seconds) * length * seconds
assert actual is not None and math.isclose(actual, expected, rel_tol=a.rtol, abs_tol=0.0), (actual, expected)
