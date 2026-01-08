from dataclasses import dataclass

import numpy as np


@dataclass
class TimeSeries:
    values: list[float]

    def __post_init__(self) -> None:
        arr = np.asarray(self.values)
        if arr.size == 0:
            raise ValueError("TimeSeries cannot be empty")
        if np.any(arr < 0):
            raise ValueError("TimeSeries contains negative values")
        if not np.all(np.isfinite(arr)):
            raise ValueError("TimeSeries contains non-finite values")

    def __getitem__(self, t: int) -> float:
        return self.values[t]

    def __len__(self) -> int:
        return len(self.values)
