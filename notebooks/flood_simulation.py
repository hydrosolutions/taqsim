import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from dataclasses import dataclass

    from taqsim.common import EVAPORATION, SEEPAGE, LossReason
    from taqsim.node import (
        Source,
        Storage,
        Splitter,
        Demand,
        Sink,
        TimeSeries,
        WaterReceived,
        WaterDistributed,
        DeficitRecorded,
    )
    from taqsim.edge import Edge, EdgeLossRule, WaterDelivered, WaterLost
    from taqsim.common import CAPACITY_EXCEEDED
    return (
        CAPACITY_EXCEEDED,
        DeficitRecorded,
        Demand,
        EVAPORATION,
        Edge,
        LossReason,
        SEEPAGE,
        Sink,
        Source,
        Splitter,
        Storage,
        TimeSeries,
        WaterDelivered,
        WaterDistributed,
        WaterLost,
        dataclass,
        mo,
    )


@app.cell
def _(mo):
    mo.md("""
    # Flood Simulation: Dam-Farm-City Network

    A 6-node water system demonstrating flood risk when dam releases exceed city channel capacity.

    ## Network Topology
    ```
    Source → Dam → Turbine → Splitter → Farm (irrigation)↘→ City (main river, capacity=100)
    ```

    ## Key Constraints
    - **City Channel Capacity**: 100 units/second (flood threshold)
    - **Farm**: Takes only what it needs (up to demand)
    - **City**: Receives everything that's left over — vulnerable to flooding
    """)
    return


@app.cell
def _(EVAPORATION, LossReason, SEEPAGE, TimeSeries, dataclass):
    # Custom Strategies

    @dataclass
    class EqualSplit:
        """Split water equally among targets."""

        def split(self, amount: float, targets: list[str], t: int) -> dict[str, float]:
            if not targets:
                return {}
            share = amount / len(targets)
            return {target: share for target in targets}


    @dataclass
    class SingleTarget:
        """Pass 100% of water to a single target (pass-through)."""

        def split(self, amount: float, targets: list[str], t: int) -> dict[str, float]:
            if not targets:
                return {}
            return {targets[0]: amount}


    @dataclass
    class FarmFirstSplit:
        """Give farm its demand first, city gets the rest."""

        farm_demand: TimeSeries

        def split(self, amount: float, targets: list[str], t: int) -> dict[str, float]:
            farm_id = targets[0]  # "farm"
            city_id = targets[1]  # "city"

            farm_need = self.farm_demand[t]
            farm_allocation = min(amount, farm_need)
            city_allocation = amount - farm_allocation

            return {farm_id: farm_allocation, city_id: city_allocation}


    @dataclass
    class FixedRelease:
        """Release a fixed fraction of storage."""

        fraction: float = 0.3

        def release(
            self,
            storage: float,
            dead_storage: float,
            capacity: float,
            inflow: float,
            t: int,
            dt: float,
        ) -> float:
            return storage * self.fraction


    @dataclass
    class SimpleLoss:
        """Simple evaporation and seepage losses."""

        evap_rate: float = 0.01
        seep_rate: float = 0.005

        def calculate(
            self, storage: float, capacity: float, t: int, dt: float
        ) -> dict[LossReason, float]:
            return {
                EVAPORATION: storage * self.evap_rate * dt,
                SEEPAGE: storage * self.seep_rate * dt,
            }


    @dataclass
    class ZeroEdgeLoss:
        """No losses during transport."""

        def calculate(
            self, flow: float, capacity: float, t: int, dt: float
        ) -> dict[LossReason, float]:
            return {}


    @dataclass
    class TransportLoss:
        """Fixed fraction loss during transport."""

        fraction: float = 0.05

        def calculate(
            self, flow: float, capacity: float, t: int, dt: float
        ) -> dict[LossReason, float]:
            return {SEEPAGE: flow * self.fraction}
    return (
        EqualSplit,
        FarmFirstSplit,
        FixedRelease,
        SimpleLoss,
        SingleTarget,
        TransportLoss,
        ZeroEdgeLoss,
    )


@app.cell
def _(mo):
    mo.md("""
    ## System Parameters

    Configure the simulation parameters below.
    """)
    return


@app.cell
def _(mo):
    # UI Controls
    dam_release_slider = mo.ui.slider(
        start=0.1, stop=0.8, step=0.05, value=0.3, label="Dam Release Fraction"
    )
    farm_demand_slider = mo.ui.slider(
        start=10, stop=80, step=5, value=30, label="Farm Demand (units/s)"
    )
    initial_storage_slider = mo.ui.slider(
        start=100, stop=900, step=50, value=500, label="Initial Dam Storage"
    )

    mo.hstack(
        [dam_release_slider, farm_demand_slider, initial_storage_slider],
        justify="start",
        gap=2,
    )
    return dam_release_slider, farm_demand_slider, initial_storage_slider


@app.cell
def _(
    Demand,
    EqualSplit,
    FarmFirstSplit,
    FixedRelease,
    SimpleLoss,
    SingleTarget,
    Sink,
    Source,
    Splitter,
    Storage,
    TimeSeries,
    dam_release_slider,
    farm_demand_slider,
    initial_storage_slider,
):
    # River inflow pattern (monthly, varying)
    river_inflow = TimeSeries([80, 100, 150, 200, 180, 120, 90, 70, 60, 80, 100, 120])

    # Farm demand (constant for simplicity)
    farm_demand = TimeSeries([farm_demand_slider.value] * 12)

    # City drinking water requirement
    city_requirement = TimeSeries([20] * 12)

    # === CREATE NODES ===

    # 1. River Source
    river = Source(
        id="river",
        inflow=river_inflow,
        targets=["dam"],
        split_strategy=SingleTarget(),
    )

    # 2. Dam (Storage)
    dam = Storage(
        id="dam",
        capacity=1000.0,
        initial_storage=initial_storage_slider.value,
        release_rule=FixedRelease(fraction=dam_release_slider.value),
        loss_rule=SimpleLoss(),
        split_strategy=SingleTarget(),
        targets=["turbine"],
    )

    # 3. Turbine (pass-through Splitter for power tracking)
    turbine = Splitter(
        id="turbine",
        targets=["junction"],
        split_strategy=SingleTarget(),
    )

    # 4. Junction (splits between farm and city)
    junction = Splitter(
        id="junction",
        targets=["farm", "city"],
        split_strategy=FarmFirstSplit(farm_demand=farm_demand),
    )

    # 5. Farm (Demand node)
    farm = Demand(
        id="farm",
        requirement=farm_demand,
        targets=[],  # terminal for this branch
        split_strategy=EqualSplit(),
    )

    # 6. City (Sink - receives remaining flow)
    city = Sink(id="city")

    nodes = {
        "river": river,
        "dam": dam,
        "turbine": turbine,
        "junction": junction,
        "farm": farm,
        "city": city,
    }
    return city_requirement, nodes


@app.cell
def _(Edge, TransportLoss, ZeroEdgeLoss):
    # === CREATE EDGES ===

    # River to Dam (high capacity, some seepage)
    e_river_dam = Edge(
        id="river_to_dam",
        source="river",
        target="dam",
        capacity=500.0,
        loss_rule=TransportLoss(fraction=0.02),
    )

    # Dam to Turbine (high capacity, no loss)
    e_dam_turbine = Edge(
        id="dam_to_turbine",
        source="dam",
        target="turbine",
        capacity=500.0,
        loss_rule=ZeroEdgeLoss(),
    )

    # Turbine to Junction (high capacity, no loss)
    e_turbine_junction = Edge(
        id="turbine_to_junction",
        source="turbine",
        target="junction",
        capacity=500.0,
        loss_rule=ZeroEdgeLoss(),
    )

    # Junction to Farm (high capacity canal)
    e_junction_farm = Edge(
        id="junction_to_farm",
        source="junction",
        target="farm",
        capacity=200.0,
        loss_rule=TransportLoss(fraction=0.03),
    )

    # Junction to City (LIMITED CAPACITY - flood constraint!)
    e_junction_city = Edge(
        id="junction_to_city",
        source="junction",
        target="city",
        capacity=100.0,  # FLOOD THRESHOLD
        loss_rule=ZeroEdgeLoss(),
    )

    edges = {
        "river_to_dam": e_river_dam,
        "dam_to_turbine": e_dam_turbine,
        "turbine_to_junction": e_turbine_junction,
        "junction_to_farm": e_junction_farm,
        "junction_to_city": e_junction_city,
    }
    return (edges,)


@app.cell
def _(mo):
    mo.md("""
    ## Simulation

    Running the water balance simulation for 12 months.
    """)
    return


@app.cell
def _(CAPACITY_EXCEEDED, DeficitRecorded, WaterDistributed, WaterLost, edges, nodes):
    def simulate_timestep(t: int, dt: float = 1.0):
        """Simulate one timestep of the water system."""

        # 1. Source generates and distributes
        nodes["river"].update(t, dt)

        # 2. Transfer river → dam via edge
        river_dist = nodes["river"].events_of_type(WaterDistributed)
        river_to_dam = sum(e.amount for e in river_dist if e.target_id == "dam" and e.t == t)
        edges["river_to_dam"].receive(river_to_dam, t)
        dam_inflow = edges["river_to_dam"].update(t, dt)

        # 3. Dam receives, updates (store, lose, release)
        nodes["dam"].receive(dam_inflow, "river_to_dam", t)
        nodes["dam"].update(t, dt)

        # 4. Transfer dam → turbine
        dam_dist = nodes["dam"].events_of_type(WaterDistributed)
        dam_to_turbine = sum(e.amount for e in dam_dist if e.target_id == "turbine" and e.t == t)
        edges["dam_to_turbine"].receive(dam_to_turbine, t)
        turbine_inflow = edges["dam_to_turbine"].update(t, dt)

        # 5. Turbine receives and passes through
        nodes["turbine"].receive(turbine_inflow, "dam_to_turbine", t)
        nodes["turbine"].update(t, dt)

        # 6. Transfer turbine → junction
        turbine_dist = nodes["turbine"].events_of_type(WaterDistributed)
        turbine_to_junction = sum(e.amount for e in turbine_dist if e.target_id == "junction" and e.t == t)
        edges["turbine_to_junction"].receive(turbine_to_junction, t)
        junction_inflow = edges["turbine_to_junction"].update(t, dt)

        # 7. Junction receives and splits
        nodes["junction"].receive(junction_inflow, "turbine_to_junction", t)
        nodes["junction"].update(t, dt)

        # 8. Transfer junction → farm
        junction_dist = nodes["junction"].events_of_type(WaterDistributed)
        junction_to_farm = sum(e.amount for e in junction_dist if e.target_id == "farm" and e.t == t)
        edges["junction_to_farm"].receive(junction_to_farm, t)
        farm_inflow = edges["junction_to_farm"].update(t, dt)

        # 9. Transfer junction → city
        junction_to_city = sum(e.amount for e in junction_dist if e.target_id == "city" and e.t == t)
        edges["junction_to_city"].receive(junction_to_city, t)
        city_inflow = edges["junction_to_city"].update(t, dt)

        # 10. Farm and City receive
        nodes["farm"].receive(farm_inflow, "junction_to_farm", t)
        nodes["farm"].update(t, dt)
        nodes["city"].receive(city_inflow, "junction_to_city", t)
        nodes["city"].update(t, dt)

    # Run simulation for 12 months
    results = {
        "month": [],
        "river_inflow": [],
        "dam_storage": [],
        "dam_release": [],
        "turbine_flow": [],
        "farm_allocation": [],
        "city_allocation": [],
        "farm_satisfied": [],
        "farm_deficit": [],
        "city_flood_excess": [],
    }

    for t in range(12):
        # Record pre-update values
        results["month"].append(t + 1)
        results["dam_storage"].append(nodes["dam"].storage)

        # Simulate
        simulate_timestep(t, dt=1.0)

        # Extract results from events
        river_gen = [e.amount for e in nodes["river"].events if hasattr(e, "amount") and e.t == t]
        results["river_inflow"].append(sum(river_gen))

        dam_released = [e.amount for e in nodes["dam"].events_of_type(WaterDistributed) if e.t == t]
        results["dam_release"].append(sum(dam_released))

        turbine_flow = [e.amount for e in nodes["turbine"].events_of_type(WaterDistributed) if e.t == t]
        results["turbine_flow"].append(sum(turbine_flow))

        farm_alloc = [e.amount for e in nodes["junction"].events_of_type(WaterDistributed)
                      if e.target_id == "farm" and e.t == t]
        results["farm_allocation"].append(sum(farm_alloc))

        city_alloc = [e.amount for e in nodes["junction"].events_of_type(WaterDistributed)
                      if e.target_id == "city" and e.t == t]
        results["city_allocation"].append(sum(city_alloc))

        # Farm satisfaction
        farm_consumed = [e.amount for e in nodes["farm"].events if hasattr(e, "amount") and e.t == t
                        and type(e).__name__ == "WaterConsumed"]
        farm_deficits = nodes["farm"].events_of_type(DeficitRecorded)
        farm_def_t = [e.deficit for e in farm_deficits if e.t == t]
        results["farm_satisfied"].append(sum(farm_consumed))
        results["farm_deficit"].append(sum(farm_def_t))

        # City flood excess
        city_losses = edges["junction_to_city"].events_of_type(WaterLost)
        city_excess_t = [e.amount for e in city_losses if e.t == t and e.reason == CAPACITY_EXCEEDED]
        results["city_flood_excess"].append(sum(city_excess_t))

    results
    return (results,)


@app.cell
def _(mo, results):
    # Summary statistics
    total_flood = sum(results["city_flood_excess"])
    flood_months = sum(1 for x in results["city_flood_excess"] if x > 0)
    total_farm_deficit = sum(results["farm_deficit"])

    summary_md = f"""
    ## Results Summary

    | Metric | Value |
    |--------|-------|
    | Total Flood Excess | **{total_flood:.1f}** units |
    | Months with Flooding | **{flood_months}** / 12 |
    | Total Farm Deficit | **{total_farm_deficit:.1f}** units |
    | Final Dam Storage | **{results['dam_storage'][-1]:.1f}** units |
    """

    if flood_months > 0:
        summary_md += "\n**⚠️ FLOOD WARNING**: City channel capacity exceeded!"

    mo.md(summary_md)
    return


@app.cell
def _(mo, results):
    import altair as alt
    import pandas as pd

    df = pd.DataFrame(results)

    # Flow chart
    flow_data = df.melt(
        id_vars=["month"],
        value_vars=["river_inflow", "dam_release", "farm_allocation", "city_allocation"],
        var_name="Flow Type",
        value_name="Flow (units/s)",
    )

    flow_chart = (
        alt.Chart(flow_data)
        .mark_line(point=True)
        .encode(
            x=alt.X("month:O", title="Month"),
            y=alt.Y("Flow (units/s):Q"),
            color=alt.Color("Flow Type:N"),
            strokeDash=alt.StrokeDash("Flow Type:N"),
        )
        .properties(width=600, height=300, title="Water Flows Through the System")
    )

    mo.md("## Flow Analysis")
    return alt, df, flow_chart


@app.cell
def _(flow_chart):
    flow_chart
    return


@app.cell
def _(alt, df, mo):
    # Dam storage chart
    storage_chart = (
        alt.Chart(df)
        .mark_area(opacity=0.6, color="steelblue")
        .encode(
            x=alt.X("month:O", title="Month"),
            y=alt.Y("dam_storage:Q", title="Dam Storage (units)"),
        )
        .properties(width=600, height=200, title="Dam Storage Over Time")
    )

    mo.vstack([mo.md("## Dam Storage"), storage_chart])
    return


@app.cell
def _(alt, df, mo):
    # Flood analysis
    flood_base = alt.Chart(df).encode(x=alt.X("month:O", title="Month"))

    city_flow = flood_base.mark_bar(color="lightblue").encode(
        y=alt.Y("city_allocation:Q", title="City Flow (units/s)")
    )

    flood_excess = flood_base.mark_bar(color="red").encode(
        y=alt.Y("city_flood_excess:Q", title="Flood Excess")
    )

    threshold = (
        alt.Chart(df)
        .mark_rule(color="darkred", strokeDash=[5, 5])
        .encode(y=alt.datum(100))
    )

    flood_chart = (
        alt.layer(city_flow, flood_excess, threshold)
        .resolve_scale(y="independent")
        .properties(width=600, height=250, title="City Channel: Flow vs Capacity (100)")
    )

    mo.vstack([mo.md("## Flood Risk Analysis"), flood_chart])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Water Balance Equation

    At the Splitter (junction), the water balance is:

    $$Flow_{City} = Release_{Dam} - Flow_{Farm}$$

    Where:
    - **Farm** takes: $\min(Demand_{Farm}, Available)$
    - **City** takes: $Remaining = Available - Farm_{allocation}$

    If $Flow_{City} > 100$, the city floods and excess water is lost.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Detailed Results
    """)
    return


@app.cell
def _(df):
    df.round(2)
    return


if __name__ == "__main__":
    app.run()
