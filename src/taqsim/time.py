from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta
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


def _add_months(d: date, months: int) -> date:
    total = d.year * 12 + d.month - 1 + months
    year, month = divmod(total, 12)
    month += 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(d.day, last_day))


def time_index(start: date, frequency: Frequency, n: int) -> tuple[date, ...]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return ()
    match frequency:
        case Frequency.DAILY:
            return tuple(start + timedelta(days=i) for i in range(n))
        case Frequency.WEEKLY:
            return tuple(start + timedelta(weeks=i) for i in range(n))
        case Frequency.MONTHLY:
            return tuple(_add_months(start, i) for i in range(n))
        case Frequency.YEARLY:
            return tuple(_add_months(start, i * 12) for i in range(n))
