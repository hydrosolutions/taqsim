from .builtins import deficit, spill
from .lift import lift
from .objective import Direction, Objective
from .registry import ObjectiveRegistry
from .trace import HasTimestep, Trace

minimize = ObjectiveRegistry("minimize")
maximize = ObjectiveRegistry("maximize")

minimize.register("spill", spill)
minimize.register("deficit", deficit)

__all__ = [
    "Direction",
    "HasTimestep",
    "Objective",
    "ObjectiveRegistry",
    "Trace",
    "deficit",
    "lift",
    "maximize",
    "minimize",
    "spill",
]
