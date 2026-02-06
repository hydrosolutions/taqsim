from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Frequency(IntEnum):
    DAILY = 365
    WEEKLY = 52
    MONTHLY = 12
    YEARLY = 1

    @staticmethod
    def scale(value: float, from_freq: Frequency, to_freq: Frequency) -> float:
        return value * from_freq / to_freq


@dataclass(frozen=True, slots=True)
class Timestep:
    index: int
    frequency: Frequency

    def __index__(self) -> int:
        return self.index

    def __int__(self) -> int:
        return self.index

    def __mod__(self, other: int) -> int:
        return self.index % other

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Timestep):
            return self.index == other.index and self.frequency == other.frequency
        if isinstance(other, int):
            return self.index == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.index)

    def scale(self, value: float, from_freq: Frequency, to_freq: Frequency | None = None) -> float:
        if to_freq is None:
            to_freq = self.frequency
        return Frequency.scale(value, from_freq, to_freq)
