# Taqsim

Water system simulation framework using event sourcing.

## Overview

Taqsim models water systems as directed acyclic graphs (DAGs) of nodes connected by edges. Water flows from sources through the network, getting stored, consumed, split, and eventually reaching sinks. All operations emit events, providing a complete audit trail of the simulation.

## Features

- **Six node types**: Source, Storage, Demand, Sink, Splitter, PassThrough
- **Configurable behavior**: Pluggable strategies for release rules, loss calculations, and flow splitting
- **Event sourcing**: Every water movement recorded as queryable events
- **Validation**: Automatic network structure validation (acyclic, connected, proper terminals)
- **Visualization**: Geographic network plotting with `system.visualize()`

## Installation

```sh
git clone https://github.com/hydrosolutions/taqsim.git
cd taqsim
uv sync
```

## Quick Example

```python
from taqsim import WaterSystem, Source, Demand, Sink, Edge, TimeSeries

# Simple loss rule that applies no losses
class NoLoss:
    def calculate(self, flow, capacity, t, dt):
        return {}

# Define nodes
source = Source(id="river", inflow=TimeSeries([100.0] * 12))
farm = Demand(id="farm", requirement=TimeSeries([30.0] * 12))
outlet = Sink(id="outlet")

# Build system
system = WaterSystem(dt=1)
system.add_node(source)
system.add_node(farm)
system.add_node(outlet)

# Connect with edges
system.add_edge(Edge(id="e1", source="river", target="farm", capacity=200, loss_rule=NoLoss()))
system.add_edge(Edge(id="e2", source="farm", target="outlet", capacity=200, loss_rule=NoLoss()))

# Simulate
system.simulate(timesteps=12)

# Query results via events
from taqsim.node import WaterReceived, DeficitRecorded

total_received = sum(e.amount for e in outlet.events_of_type(WaterReceived))
deficits = farm.events_of_type(DeficitRecorded)
```

## Documentation

- [Node Architecture](documentation/nodes/01_architecture.md)
- [Node Types](documentation/nodes/06_node_types.md)
- [Edge Architecture](documentation/edges/01_architecture.md)
- [System Architecture](documentation/system/01_architecture.md)

## Contact

For questions or suggestions, please open an issue or contact the maintainers at [hydrosolutions](mailto:info@hydrosolutions.ch).
