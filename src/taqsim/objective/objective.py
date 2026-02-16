from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from taqsim.system import WaterSystem

Direction = Literal["minimize", "maximize"]


@dataclass(frozen=True, slots=True)
class Objective:
    name: str
    direction: Direction
    evaluate: Callable[["WaterSystem"], float]
