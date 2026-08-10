# Taqsim

Event-sourced framework for simulating and optimizing water allocation across directed network models.

## Overview

Taqsim represents a water system as a directed acyclic graph (DAG). Water moves from sources through storage, demand, routing, and splitting nodes to sinks, while simulation events provide a complete audit trail. Declarative objectives and a multi-objective optimizer expose trade-offs among operating policies without embedding policy decisions in the simulation engine.

## Maintenance Status

🟢 **Active Development**

This repository is part of an ongoing project and actively maintained.

## Features

- **Water-network simulation**: Source, Storage, Demand, Sink, Splitter, PassThrough, and Reach nodes
- **Configurable behavior**: Pluggable release, loss, routing, and flow-splitting strategies
- **Event sourcing**: Queryable events record water generation, movement, consumption, loss, deficits, and storage
- **Validation**: Structural and time-series checks for connected, acyclic networks with valid terminals
- **Optimization**: Declarative objectives and NSGA-II multi-objective optimization with parameter constraints
- **Interchange and analysis**: JSON serialization, parameter introspection, time indexing, and geographic visualization

## Design Philosophy

Taqsim follows a single rule: **expose everything, decide nothing**.

The simulation engine routes water, records events, and exposes all state for external analysis. It never interprets data or makes policy decisions. Optimization, visualization, and decision support are separate concerns built on top of taqsim's transparent foundation.

See [Design Philosophy](src/taqsim/documentation/00_philosophy.md) for details.

## Installation

```sh
git clone https://github.com/hydrosolutions/taqsim.git
cd taqsim
uv sync
```

## Quick Example

```python
from taqsim import Demand, Frequency, Sink, Source, TimeSeries, WaterSystem
from taqsim.node import DeficitRecorded, WaterReceived

source = Source(id="river", inflow=TimeSeries([100.0] * 12))
farm = Demand(id="farm", requirement=TimeSeries([30.0] * 12))
outlet = Sink(id="outlet")

system = WaterSystem(frequency=Frequency.MONTHLY)
for node in (source, farm, outlet):
    system.add_node(node)

system.connect("river", "farm").connect("farm", "outlet")
system.simulate(timesteps=12)

total_received = sum(event.amount for event in outlet.events_of_type(WaterReceived))
deficits = farm.events_of_type(DeficitRecorded)
```

## Documentation

- [Design philosophy](src/taqsim/documentation/00_philosophy.md)
- [Node types](src/taqsim/documentation/nodes/06_node_types.md)
- [System architecture](src/taqsim/documentation/system/01_architecture.md)
- [Objectives](src/taqsim/documentation/objective/01_overview.md)
- [Multi-objective optimization](src/taqsim/documentation/optimization/01_overview.md)
- [JSON interchange](src/taqsim/documentation/system/04_json.md)
- [Consuming the packaged documentation](src/taqsim/documentation/30_consuming_docs.md)

## Contact

For questions or suggestions, please open an issue or contact the maintainers at [hydrosolutions](mailto:info@hydrosolutions.ch).
