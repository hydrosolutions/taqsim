#!/usr/bin/env python3
"""Explore system configurations to find interesting Pareto trade-offs.

New topology with genuine competing objectives:
- river → reservoir → turbine → city (PassThrough) → splitter → irrigation + thermal

Four objectives:
1. Hydropower (maximize) - wants high release
2. City flood (minimize) - wants release ≤ city capacity
3. Irrigation deficit (minimize) - consumptive demand
4. Thermal deficit (minimize) - non-consumptive cooling demand
"""

import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import numpy as np

from taqsim import (
    Demand,
    Edge,
    LossReason,
    Objective,
    Ordered,
    PassThrough,
    Sink,
    Source,
    Splitter,
    Storage,
    Strategy,
    TimeSeries,
    WaterSystem,
    optimize,
)
from taqsim.node.events import DeficitRecorded, WaterPassedThrough, WaterReleased, WaterSpilled, WaterStored

TIMESTEPS = 120
LOGS_DIR = Path(__file__).parent.parent / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# --- Time Series Generation ---


def generate_inflows(n_years: int, mean_annual: float, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    seasonal = [0.5, 0.6, 0.8, 1.2, 1.6, 1.8, 1.5, 1.0, 0.7, 0.5, 0.4, 0.4]
    inflows = []
    for year in range(n_years):
        year_factor = rng.uniform(0.8, 1.2)
        for month, s in enumerate(seasonal):
            noise = rng.uniform(0.9, 1.1)
            extreme = 1.8 if rng.random() < 0.08 else 1.0
            inflows.append(mean_annual * s * year_factor * noise * extreme)
    return inflows


def generate_irrigation_demand(n_years: int, peak_demand: float, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    seasonal = [0.0, 0.0, 0.2, 0.6, 1.0, 1.4, 1.5, 1.2, 0.7, 0.2, 0.0, 0.0]
    demand = []
    for year in range(n_years):
        for month, s in enumerate(seasonal):
            noise = rng.uniform(0.9, 1.1)
            demand.append(peak_demand * s * noise)
    return demand


def generate_thermal_demand(n_years: int, base_demand: float, seed: int) -> list[float]:
    """Generate thermal plant cooling demand - higher in summer (more cooling needed)."""
    rng = np.random.default_rng(seed)
    seasonal = [0.7, 0.7, 0.8, 0.9, 1.1, 1.3, 1.4, 1.3, 1.1, 0.9, 0.8, 0.7]
    demand = []
    for year in range(n_years):
        for month, s in enumerate(seasonal):
            noise = rng.uniform(0.95, 1.05)
            demand.append(base_demand * s * noise)
    return demand


# --- Strategies ---


def volume_to_head(
    volume: float, v_dead: float = 10.0, v_max: float = 150.0, h_dead: float = 20.0, h_max: float = 100.0
) -> float:
    v_clamped = max(v_dead, min(v_max, volume))
    return h_dead + (v_clamped - v_dead) * (h_max - h_dead) / (v_max - v_dead)


@dataclass(frozen=True)
class SLOPRelease(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("h1", "h2", "w", "m1", "m2")
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
        "h1": (20.0, 100.0),
        "h2": (20.0, 100.0),
        "w": (0.0, 80.0),
        "m1": (0.01, 2.0),
        "m2": (0.01, 2.0),
    }
    __constraints__: ClassVar[tuple] = (Ordered(low="h1", high="h2"),)
    __time_varying__: ClassVar[tuple[str, ...]] = ("h1", "h2", "w", "m1", "m2")
    __cyclical__: ClassVar[tuple[str, ...]] = ("h1", "h2", "w", "m1", "m2")

    h1: tuple[float, ...] = (40.0,) * 12
    h2: tuple[float, ...] = (70.0,) * 12
    w: tuple[float, ...] = (40.0,) * 12
    m1: tuple[float, ...] = (0.5,) * 12
    m2: tuple[float, ...] = (0.8,) * 12

    def release(self, node: Storage, inflow: float, t: int, dt: float) -> float:
        head = volume_to_head(node.storage)
        month = t % 12
        h1_t, h2_t, w_t, m1_t, m2_t = self.h1[month], self.h2[month], self.w[month], self.m1[month], self.m2[month]

        if head < h1_t:
            release = max(0.0, w_t - m1_t * (h1_t - head))
        elif head > h2_t:
            release = w_t + m2_t * (head - h2_t)
        else:
            release = w_t

        available = max(0.0, node.storage - node.dead_storage)
        return min(release * dt, available)


@dataclass(frozen=True)
class SeasonalRatio(Strategy):
    """Time-varying split ratio between irrigation and thermal."""

    __params__: ClassVar[tuple[str, ...]] = ("irrigation_fraction",)
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"irrigation_fraction": (0.1, 0.9)}
    __time_varying__: ClassVar[tuple[str, ...]] = ("irrigation_fraction",)
    __cyclical__: ClassVar[tuple[str, ...]] = ("irrigation_fraction",)

    irrigation_fraction: tuple[float, ...] = (0.5,) * 12

    def split(self, node: Splitter, amount: float, t: int) -> dict[str, float]:
        frac = self.irrigation_fraction[t % 12]
        return {"irrigation": amount * frac, "thermal_plant": amount * (1.0 - frac)}


@dataclass(frozen=True)
class ZeroEdgeLoss:
    def calculate(self, edge: Edge, flow: float, t: int, dt: float) -> dict[LossReason, float]:
        return {}


@dataclass(frozen=True)
class ZeroStorageLoss:
    def calculate(self, node: Storage, t: int, dt: float) -> dict[LossReason, float]:
        return {}


# --- Objectives ---


def hydropower_objective(
    reservoir_id: str, turbine_id: str, initial_storage: float, efficiency: float = 0.85
) -> Objective:
    def evaluate(system: WaterSystem) -> float:
        turbine = system.nodes[turbine_id]
        reservoir = system.nodes[reservoir_id]
        flow_trace = turbine.trace(WaterPassedThrough)
        stored_trace = reservoir.trace(WaterStored)
        released_trace = reservoir.trace(WaterReleased)
        net_change = stored_trace - released_trace
        storage_trace = net_change.cumsum(initial=initial_storage)
        head_trace = storage_trace.map(lambda v: volume_to_head(v))
        power_trace = flow_trace * head_trace * (9810 * efficiency / 1e9)
        return power_trace.sum()

    return Objective(name="hydropower", direction="maximize", evaluate=evaluate)


def flood_objective(node_id: str) -> Objective:
    """Minimize flood/spill at a PassThrough node."""

    def evaluate(system: WaterSystem) -> float:
        return system.nodes[node_id].trace(WaterSpilled).sum()

    return Objective(name="city_flood", direction="minimize", evaluate=evaluate)


def deficit_objective(node_id: str, name: str) -> Objective:
    """Minimize deficit at a Demand node."""

    def evaluate(system: WaterSystem) -> float:
        return system.nodes[node_id].trace(DeficitRecorded, field="deficit").sum()

    return Objective(name=name, direction="minimize", evaluate=evaluate)


# --- System Builder ---


@dataclass
class Config:
    city_capacity: float
    irrigation_peak: float
    thermal_base: float
    inflow_scale: float
    seed: int

    def to_dict(self) -> dict:
        return {
            "city_capacity": self.city_capacity,
            "irrigation_peak": self.irrigation_peak,
            "thermal_base": self.thermal_base,
            "inflow_scale": self.inflow_scale,
            "seed": self.seed,
        }


def build_system(config: Config) -> WaterSystem:
    base_inflow = 100.0

    inflows = generate_inflows(10, base_inflow * config.inflow_scale, config.seed)
    irrigation_demand = generate_irrigation_demand(10, config.irrigation_peak, config.seed + 1)
    thermal_demand = generate_thermal_demand(10, config.thermal_base, config.seed + 2)

    system = WaterSystem(dt=1.0)

    # Nodes
    system.add_node(Source(id="river", inflow=TimeSeries(inflows)))
    system.add_node(
        Storage(
            id="reservoir",
            capacity=150.0,
            dead_storage=10.0,
            initial_storage=75.0,
            release_policy=SLOPRelease(),
            loss_rule=ZeroStorageLoss(),
        )
    )
    system.add_node(PassThrough(id="turbine", capacity=60.0))
    system.add_node(PassThrough(id="city", capacity=config.city_capacity))  # Bottleneck!
    system.add_node(Splitter(id="splitter", split_policy=SeasonalRatio()))
    system.add_node(
        Demand(
            id="irrigation",
            requirement=TimeSeries(irrigation_demand),
            consumption_fraction=1.0,  # Fully consumptive
            efficiency=0.85,
        )
    )
    system.add_node(
        Demand(
            id="thermal_plant",
            requirement=TimeSeries(thermal_demand),
            consumption_fraction=0.0,  # Non-consumptive (cooling)
            efficiency=1.0,
        )
    )
    system.add_node(Sink(id="irrigation_sink"))
    system.add_node(Sink(id="thermal_sink"))

    # Edges
    zero_loss = ZeroEdgeLoss()
    edges = [
        ("e_river_res", "river", "reservoir", 200.0),
        ("e_res_turb", "reservoir", "turbine", 80.0),
        ("e_turb_city", "turbine", "city", 80.0),
        ("e_city_split", "city", "splitter", 80.0),
        ("e_split_irr", "splitter", "irrigation", 80.0),
        ("e_split_therm", "splitter", "thermal_plant", 80.0),
        ("e_irr_sink", "irrigation", "irrigation_sink", 80.0),
        ("e_therm_sink", "thermal_plant", "thermal_sink", 80.0),
    ]
    for id_, src, tgt, cap in edges:
        system.add_edge(Edge(id=id_, source=src, target=tgt, capacity=cap, loss_rule=zero_loss))

    system.validate()
    return system


# --- Exploration ---


def compute_spread(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    min_v, max_v = min(values), max(values)
    if max_v == 0:
        return 0.0
    return (max_v - min_v) / max_v  # Normalized spread


def evaluate_config(config: Config, pop_size: int = 50, generations: int = 30) -> dict:
    system = build_system(config)

    objectives = [
        hydropower_objective("reservoir", "turbine", initial_storage=75.0),
        flood_objective("city"),
        deficit_objective("irrigation", "irrigation_deficit"),
        deficit_objective("thermal_plant", "thermal_deficit"),
    ]

    result = optimize(
        system=system,
        objectives=objectives,
        timesteps=TIMESTEPS,
        pop_size=pop_size,
        generations=generations,
        seed=config.seed,
        verbose=False,
    )

    pareto_count = len(result.solutions)

    scores_by_obj: dict[str, list[float]] = {}
    for sol in result.solutions:
        for name, val in sol.scores.items():
            scores_by_obj.setdefault(name, []).append(val)

    spreads = {name: compute_spread(vals) for name, vals in scores_by_obj.items()}
    ranges = {name: (min(vals), max(vals)) for name, vals in scores_by_obj.items()}

    # Interestingness: pareto count × average normalized spread
    avg_spread = sum(spreads.values()) / len(spreads) if spreads else 0
    interestingness = pareto_count * (1 + avg_spread * 10)

    return {
        "config": config.to_dict(),
        "pareto_count": pareto_count,
        "spreads": spreads,
        "ranges": ranges,
        "interestingness": interestingness,
    }


def sample_config(rng: random.Random, trial: int) -> Config:
    return Config(
        city_capacity=rng.uniform(25.0, 50.0),  # Sweet spot for trade-offs
        irrigation_peak=rng.uniform(25.0, 40.0),  # Peak irrigation demand
        thermal_base=rng.uniform(20.0, 35.0),  # Base thermal cooling demand
        inflow_scale=rng.uniform(0.8, 1.2),  # Moderate variation
        seed=rng.randint(0, 100000),
    )


def run_exploration(n_trials: int = 100, seed: int = 42) -> None:
    rng = random.Random(seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"exploration_{timestamp}.jsonl"

    log.info(f"Starting exploration: {n_trials} trials")
    log.info(f"Logging to: {log_file}")
    log.info("Topology: river → reservoir → turbine → city → splitter → irrigation + thermal")
    log.info("Objectives: hydropower (max), city_flood (min), irrigation_deficit (min), thermal_deficit (min)")

    best_interestingness = 0.0
    best_result = None

    for trial in range(n_trials):
        config = sample_config(rng, trial)

        try:
            start = time.time()
            result = evaluate_config(config)
            elapsed = time.time() - start

            result["trial"] = trial
            result["elapsed_seconds"] = round(elapsed, 1)

            with open(log_file, "a") as f:
                f.write(json.dumps(result) + "\n")

            pareto = result["pareto_count"]
            interest = result["interestingness"]

            if interest > best_interestingness:
                best_interestingness = interest
                best_result = result
                marker = " *** NEW BEST ***"
            else:
                marker = ""

            log.info(
                f"Trial {trial:3d} | pareto={pareto:3d} | interest={interest:7.1f} | "
                f"city_cap={config.city_capacity:5.1f} | irr={config.irrigation_peak:5.1f} | "
                f"therm={config.thermal_base:5.1f} | inflow={config.inflow_scale:.2f} | "
                f"{elapsed:.1f}s{marker}"
            )

        except Exception as e:
            log.error(f"Trial {trial} failed: {e}")
            continue

    log.info("=" * 80)
    log.info("Exploration complete!")
    if best_result:
        log.info(f"Best interestingness: {best_interestingness:.1f}")
        log.info(f"Best config: {json.dumps(best_result['config'], indent=2)}")
        log.info(f"Pareto count: {best_result['pareto_count']}")
        log.info(f"Spreads: {json.dumps(best_result['spreads'], indent=2)}")
        log.info(
            f"Ranges: {json.dumps({k: [round(v[0], 2), round(v[1], 2)] for k, v in best_result['ranges'].items()}, indent=2)}"
        )


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_exploration(n_trials=n_trials)
