import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    from taqsim import (
        WaterSystem,
        SupplyNode,
        DemandNode,
        SinkNode,
        Edge,
    )
    return DemandNode, Edge, SinkNode, SupplyNode, WaterSystem, mo, pd


@app.cell
def _(mo):
    mo.md("""
    # Taqsim: Simple Water System

    A minimal example: **Source → Demand → Outlet**
    """)
    return


@app.cell
def _(DemandNode, Edge, SinkNode, SupplyNode, WaterSystem):
    system = WaterSystem(dt=2629800, start_year=2020, start_month=1)

    supply = SupplyNode(
        id="Source",
        easting=0, northing=100,
        constant_supply_rate=10,
        num_time_steps=12
    )

    demand = DemandNode(
        id="Farm",
        easting=50, northing=50,
        constant_demand_rate=4,
        num_time_steps=12
    )

    sink = SinkNode(
        id="Outlet",
        easting=100, northing=0,
        constant_min_flow=2,
        num_time_steps=12
    )

    system.add_node(supply)
    system.add_node(demand)
    system.add_node(sink)

    system.add_edge(Edge(source=supply, target=demand, capacity=15))
    system.add_edge(Edge(source=demand, target=sink, capacity=15))

    system.simulate(time_steps=12)
    return demand, sink, supply


@app.cell
def _(mo):
    mo.md("""
    ## Results
    """)
    return


@app.cell
def _(demand, pd, sink, supply):
    pd.DataFrame({
        "Month": range(1, 13),
        "Supply (m³/s)": supply.supply_history,
        "Satisfied (m³/s)": demand.satisfied_consumptive_demand,
        "Unmet (m³/s)": demand.unmet_demand,
        "Outlet (m³/s)": sink.flow_history,
    })
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
