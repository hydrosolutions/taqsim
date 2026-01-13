from typing import TYPE_CHECKING

from taqsim.edge.events import WaterDelivered
from taqsim.edge.events import WaterLost as EdgeWaterLost
from taqsim.node.events import DeficitRecorded, WaterSpilled
from taqsim.node.events import WaterLost as NodeWaterLost
from taqsim.node.events import WaterReceived as NodeWaterReceived

from .objective import Objective

if TYPE_CHECKING:
    from taqsim.system import WaterSystem


def spill(node_id: str, *, priority: int = 1) -> Objective:
    def evaluate(system: "WaterSystem") -> float:
        if node_id not in system.nodes:
            raise ValueError(f"Node '{node_id}' not found")
        return system.nodes[node_id].trace(WaterSpilled).sum()

    return Objective(
        name=f"{node_id}.spill",
        direction="minimize",
        evaluate=evaluate,
        priority=priority,
    )


def deficit(node_id: str, *, priority: int = 1) -> Objective:
    def evaluate(system: "WaterSystem") -> float:
        if node_id not in system.nodes:
            raise ValueError(f"Node '{node_id}' not found")
        return system.nodes[node_id].trace(DeficitRecorded, field="deficit").sum()

    return Objective(
        name=f"{node_id}.deficit",
        direction="minimize",
        evaluate=evaluate,
        priority=priority,
    )


def delivery(target_id: str, *, priority: int = 1) -> Objective:
    def evaluate(system: "WaterSystem") -> float:
        if target_id in system.edges:
            return system.edges[target_id].trace(WaterDelivered).sum()
        if target_id in system.nodes:
            return system.nodes[target_id].trace(NodeWaterReceived).sum()
        raise ValueError(f"'{target_id}' not found as node or edge")

    return Objective(
        name=f"{target_id}.delivery",
        direction="maximize",
        evaluate=evaluate,
        priority=priority,
    )


def loss(target_id: str, *, priority: int = 1) -> Objective:
    def evaluate(system: "WaterSystem") -> float:
        if target_id in system.edges:
            return system.edges[target_id].trace(EdgeWaterLost).sum()
        if target_id in system.nodes:
            return system.nodes[target_id].trace(NodeWaterLost).sum()
        raise ValueError(f"'{target_id}' not found as node or edge")

    return Objective(
        name=f"{target_id}.loss",
        direction="minimize",
        evaluate=evaluate,
        priority=priority,
    )
