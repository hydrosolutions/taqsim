from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from ctrl_freak.population import Population
    from numpy.typing import NDArray

    from taqsim.system import WaterSystem


@dataclass(frozen=True, slots=True)
class Solution:
    scores: dict[str, float]
    parameters: dict[str, float]
    _vector: NDArray[np.float64]
    _template: WaterSystem

    def to_system(self) -> WaterSystem:
        """Returns configured but unsimulated system."""
        return self._template.with_vector(list(self._vector))


@dataclass(frozen=True, slots=True)
class OptimizeResult:
    solutions: list[Solution]
    population: Population

    def __len__(self) -> int:
        return len(self.solutions)

    def __getitem__(self, idx: int) -> Solution:
        return self.solutions[idx]
