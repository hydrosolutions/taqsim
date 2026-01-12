from dataclasses import dataclass

from taqsim.common import LossReason


@dataclass(frozen=True, slots=True)
class WaterReceived:
    amount: float
    t: int


@dataclass(frozen=True, slots=True)
class WaterLost:
    amount: float
    reason: LossReason
    t: int


@dataclass(frozen=True, slots=True)
class WaterDelivered:
    amount: float
    t: int


EdgeEvent = WaterReceived | WaterLost | WaterDelivered
