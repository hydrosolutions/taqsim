from dataclasses import dataclass

from taqsim.common import LossReason


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


@dataclass(frozen=True, slots=True)
class WaterOutput:
    """Water available for downstream. Used by single-output nodes."""

    amount: float
    t: int


@dataclass(frozen=True, slots=True)
class WaterPassedThrough:
    """Records water passing through (for analysis, e.g., turbine power)."""

    amount: float
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
    | WaterOutput
    | WaterPassedThrough
)
