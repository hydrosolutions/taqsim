from collections.abc import Callable
from functools import wraps

from .trace import Trace


def lift(fn: Callable[[float], float]) -> Callable[[Trace | float], Trace | float]:
    @wraps(fn)
    def lifted(x: Trace | float) -> Trace | float:
        if isinstance(x, Trace):
            return x.map(fn)
        return fn(x)

    return lifted
