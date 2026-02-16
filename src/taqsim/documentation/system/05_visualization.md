# Visualization

## Overview

`WaterSystem.visualize()` renders the network topology as a graph using networkx and matplotlib. Returns `(fig, ax)` for composability -- the caller controls display.

## Signature

```python
def visualize(
    self,
    *,
    show_reaches: bool = True,
    save_to: str | Path | None = None,
    figsize: tuple[float, float] = (12, 8),
    title: str | None = None,
) -> tuple[Figure, Axes]:
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `show_reaches` | `bool` | `True` | When `False`, collapses Reach nodes into labeled edges |
| `save_to` | `str \| Path \| None` | `None` | File path to save the figure |
| `figsize` | `tuple[float, float]` | `(12, 8)` | Figure size in inches |
| `title` | `str \| None` | `None` | Custom title. Auto-generates if `None` |

## Node Styling

Each node type has a distinct color and size:

| Node Type | Color | Size |
|-----------|-------|------|
| Source | Blue (#3498db) | 600 |
| Storage | Green (#27ae60) | 700 |
| Splitter | Orange (#e67e22) | 500 |
| Demand | Red (#e74c3c) | 500 |
| PassThrough | Purple (#9b59b6) | 400 |
| Reach | Brown (#8B4513) | 350 |
| Sink | Gray (#7f8c8d) | 500 |

A legend is automatically added showing only the types present in the network.

## Edges

Edges are drawn as directed arrows in light gray. When reaches are collapsed (`show_reaches=False`), the collapsed connections are drawn as dashed brown lines labeled with the reach ID.

## `show_reaches` Parameter

### `show_reaches=True` (default)

Reach nodes appear as regular nodes in the graph with their own color and connections.

### `show_reaches=False`

Reach nodes are removed from the graph. Their upstream and downstream connections are replaced by a single dashed edge labeled with the reach ID.

```
Before (show_reaches=True):
  Source -> [Reach: canal] -> Sink

After (show_reaches=False):
  Source ---canal---> Sink
```

This simplifies visualization of networks where transport details are less important than the logical topology.

## Layout Strategies

### Geographic (all nodes have locations)

When every node has a `location` set, nodes are plotted at their geographic coordinates (longitude as x, latitude as y).

### Automatic Fallback (no nodes have locations)

When no nodes have locations, `kamada_kawai_layout` is used for automatic positioning. A `UserWarning` is issued.

### Mixed (some nodes have locations)

When only some nodes have locations, the located nodes are fixed at their geographic positions and `spring_layout` positions the remaining nodes relative to them.

## Examples

### Basic visualization

```python
fig, ax = system.visualize()
plt.show()  # Caller controls display
```

### Save to file

```python
fig, ax = system.visualize(save_to="network.png")
```

### Collapsed reaches

```python
fig, ax = system.visualize(show_reaches=False, title="Logical Topology")
plt.show()
```

### Custom figure size

```python
fig, ax = system.visualize(figsize=(16, 10), title="Large Network")
plt.show()
```

### Composable with matplotlib

```python
fig, ax = system.visualize()
ax.annotate("Critical node", xy=(35.2, 31.5), fontsize=12)
fig.savefig("annotated_network.png")
```
