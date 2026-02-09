from dataclasses import dataclass, field
from typing import Any

from taqsim.time import Timestep

from .base import BaseNode
from .events import (
    WaterEnteredReach,
    WaterInTransit,
    WaterLost,
    WaterOutput,
    WaterReceived,
)
from .strategies import ReachLossRule, RoutingModel


@dataclass
class Reach(BaseNode):
    routing_model: RoutingModel | None = field(default=None)
    loss_rule: ReachLossRule | None = field(default=None)
    _routing_state: Any = field(init=False, repr=False, default=None)
    _received_this_step: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.routing_model is None:
            raise ValueError("routing_model is required")
        if self.loss_rule is None:
            raise ValueError("loss_rule is required")
        self._routing_state = self.routing_model.initial_state(self)

    @property
    def water_in_transit(self) -> float:
        return self.routing_model.storage(self._routing_state)

    def receive(self, amount: float, source_id: str, t: Timestep) -> float:
        self.record(WaterReceived(amount=amount, source_id=source_id, t=t.index))
        self._received_this_step += amount
        return amount

    def update(self, t: Timestep) -> None:
        inflow = self._received_this_step

        # 1. Record inflow entering channel
        self.record(WaterEnteredReach(amount=inflow, t=t.index))

        # 2. Route — transform (state, inflow) -> (outflow, new_state)
        outflow, new_state = self.routing_model.route(self, inflow, self._routing_state, t)
        self._routing_state = new_state

        # 3. Lose — calculate and apply losses to routed outflow
        losses = self.loss_rule.calculate(self, outflow, t)
        total_loss = sum(losses.values())

        if total_loss > outflow:
            scale = outflow / total_loss if total_loss > 0 else 0
            losses = {reason: amount * scale for reason, amount in losses.items()}
            total_loss = outflow

        for reason, amount in losses.items():
            if amount > 0:
                self.record(WaterLost(amount=amount, reason=reason, t=t.index))

        net_outflow = outflow - total_loss

        # 4. Transit snapshot
        self.record(WaterInTransit(amount=self.water_in_transit, t=t.index))

        # 5. Output downstream
        if net_outflow > 0:
            self.record_output(WaterOutput(amount=net_outflow, t=t.index))

        self._received_this_step = 0.0

    def reset(self) -> None:
        super().reset()
        self._routing_state = self.routing_model.initial_state(self)
        self._received_this_step = 0.0
