# Water System Simulation Project Summary

## Project Overview
The project is a Python-based water system simulation that models the flow of water through various types of nodes (supply, storage, demand, sink, and hydroworks) connected by edges. The system supports time-step based simulations with the ability to handle variable time steps, typically set to monthly intervals.

## Key Components
1. `WaterSystem` class: The main class that represents the entire water system. It manages the simulation, visualization, and data analysis.
2. Node classes:
  - `SupplyNode`: Represents water sources with variable supply rates
- `StorageNode`: Represents reservoirs or storage facilities with a defined capacity
- `DemandNode`: Represents points of water consumption with variable demand rates
- `SinkNode`: Represents endpoints where water exits the system
- `HydroWorks`: Represents points where water can be redistributed (combining diversion and confluence functionality)
3. `Edge` class: Represents connections between nodes, with defined capacities.

## Key Features
- Simulation of variable water flow over multiple time steps
- Water balance calculations for each node
- CSV output of simulation results
- Comprehensive visualization of the water system

## Recent Enhancements
- Time Step Handling: The system now properly accounts for the time step duration (dt) in flow calculations, especially for storage nodes.
- Variable Demand Rates: DemandNodes now support variable demand rates over time, similar to how SupplyNodes handle variable supply rates.
- Visualization: The visualize method in WaterSystem has been updated to correctly display information for nodes with variable rates.
- Water Balance Calculations: The get_water_balance_table method has been updated to handle variable demand rates and provide more accurate data.
- Seasonal Reservoir Test: A new test case has been added to simulate a system with seasonal supply and demand variations over a 10-year period.

## Sample Test Systems
Several sample test systems have been implemented:
  1. Most Simple System: Most simple system with one supply, one demand, and one sink node 
2. System with Hydroworks: System with one supply, three HydroWorks nodes, and one sink node
3. Simple System: Basic setup with one supply, one storage, two demands, and one sink.
4. Complex System: More intricate network with multiple supplies, storages, and demands.
5. Drought System: System with variable supply to simulate alternating normal and drought conditions.
6. Test Seasonal Reservoir: System wqith seasonal demand and supply variations over a 10 year period.

## Current Functionality
- Creation of water system networks with various node types
- Simulation of water flow through the system
- Detailed visualization of the water system state after simulation
- Water balance calculations and CSV output

## Next Steps

### Visualization / GUI
- [ ] Implement proper time series visualization of key system flows and volumes over time
- [ ] Improve network plotting to better represent the system layout
- [ ] Developing a user interface (GUI)

### Network
- [ ] Implementing topological sorting of the network to ensure proper flow calculations from upstream to downstream nodes so that outflow from a node is always calculated after inflow to that node

### Water Balancing
- [x] Proper flow to volume and volume to flow conversion for reservoir node water balance 
- [ ] Initialize reservoirs with initial condition of reservoir filling
- [ ] Implement reservoir spillway where V(t) <= Vmax is always true
- [ ] Each edge has a loss factor as an attribute upon which flow losses depend over distance of the edge
- [ ] Implement water level-volume relationship for reservoir nodes
- [ ] Implement evaporative losses at reservoir nodes

### Unsaturated and Saturated Zone
- [ ] Soil moisture/infiltration model
- [ ] Implementing simple representation of groundwater

### Geospatial Characteristics
- [ ] Implement node location characteristics (easting, northing) for each node
- [ ] Each edge has a length computed from the connecting node coordinates and thus has a length attribute

### Water demand
- [ ] Implement irrigation water demand at agricultural demand nodes where one enters area and crop type and irrigation water demand for Samarkand region comes out. Use Climate Explorer to get the data for relevant crops. The idea would then be to pass a dictionary of crop types and areas to the demand node and have it calculate the total irrigation water demand over time.
- [ ] Same as before, but for domestic water demand where you enter the total number of people requiring water and the demand node calculates the total water demand over time.
- [ ] Same as before, but for industrial water demand where you enter the total industrial water demand over time.

### I/O
- [ ] Implementing data import/export features to load supply and demand nodes data from .csv sources.

### Optimization
- [ ] Implementing more complex node behaviors where each node becomes a decision variable
- [ ] Incorporating optimization algorithms
- [ ] Adding support for stochastic simulations

## Issues

- Water Balance issues

- Spillway: What happens to water when upstream inflow to hydroworks structure is larger than total outflow capacity of hydroworks structure? Have to build in spillway as in the case of the reservoir and heavily penalize spillway use at any time.

## Dependencies
- NetworkX
- Matplotlib
- Pandas

This summary reflects the current state of the Water System Simulation Project, including the recent improvements to the visualization capabilities. It can be used to quickly brief someone on the project's status and capabilities.