from dataclasses import dataclass
from enum import Enum, auto


class LossReason(Enum):
    EVAPORATION = auto()
    SEEPAGE = auto()


@dataclass(frozen=True, slots=True)
class WaterGenerated:
    amount: float  # m³
    t: int  # timestep


@dataclass(frozen=True, slots=True)
class WaterReceived:
    amount: float
    source_id: str
    t: int


@dataclass(frozen=True, slots=True)
class WaterStored:
    amount: float
    t: int


@dataclass(frozen=True, slots=True)
class WaterReleased:
    amount: float
    t: int


@dataclass(frozen=True, slots=True)
class WaterLost:
    amount: float
    reason: LossReason
    t: int


@dataclass(frozen=True, slots=True)
class WaterSpilled:
    amount: float
    t: int


@dataclass(frozen=True, slots=True)
class WaterConsumed:
    amount: float
    t: int


@dataclass(frozen=True, slots=True)
class WaterDistributed:
    amount: float
    target_id: str
    t: int


@dataclass(frozen=True, slots=True)
class DeficitRecorded:
    required: float
    actual: float
    deficit: float
    t: int


NodeEvent = (
    WaterGenerated
    | WaterReceived
    | WaterStored
    | WaterReleased
    | WaterLost
    | WaterSpilled
    | WaterConsumed
    | WaterDistributed
    | DeficitRecorded
)
