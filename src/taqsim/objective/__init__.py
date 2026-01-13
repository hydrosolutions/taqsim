from .builtins import deficit, delivery, loss, spill
from .lift import lift
from .objective import Direction, Objective
from .registry import ObjectiveRegistry
from .trace import HasTimestep, Trace

minimize = ObjectiveRegistry("minimize")
maximize = ObjectiveRegistry("maximize")

minimize.register("spill", spill)
minimize.register("deficit", deficit)
minimize.register("loss", loss)

maximize.register("delivery", delivery)

__all__ = [
    "Direction",
    "HasTimestep",
    "Objective",
    "ObjectiveRegistry",
    "Trace",
    "deficit",
    "delivery",
    "lift",
    "loss",
    "maximize",
    "minimize",
    "spill",
]
