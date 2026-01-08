from dataclasses import dataclass

from taqsim.common import LossReason


@dataclass(frozen=True, slots=True)
class FlowReceived:
    amount: float
    t: int


@dataclass(frozen=True, slots=True)
class FlowLost:
    amount: float
    reason: LossReason
    t: int


@dataclass(frozen=True, slots=True)
class FlowDelivered:
    amount: float
    t: int


@dataclass(frozen=True, slots=True)
class CapacityExceeded:
    excess: float
    t: int


@dataclass(frozen=True, slots=True)
class RequirementUnmet:
    required: float
    actual: float
    deficit: float
    t: int


EdgeEvent = FlowReceived | FlowLost | FlowDelivered | CapacityExceeded | RequirementUnmet
