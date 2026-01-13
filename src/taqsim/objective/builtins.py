from typing import TYPE_CHECKING

from taqsim.node.events import DeficitRecorded, WaterSpilled

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
