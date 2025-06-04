# Water System Node Types: Detailed Reference

This document provides a comprehensive overview of each node type available in the water system simulation framework. For each node, you will find a description, main functionalities, and a detailed explanation of all key parameters and their effects.

---

## Units Overview

| Quantity                | Unit         | Description                                      |
|-------------------------|--------------|--------------------------------------------------|
| Flow rate               | m³/s         | Water flow (e.g., supply, demand, outflow)       |
| Volume                  | m³           | Storage, reservoir volume, supply/demand totals  |
| Evaporation rate        | mm           | Monthly evaporation input for reservoirs         |
| Rainfall/Precipitation  | mm           | Monthly rainfall input for runoff nodes          |
| Area                    | km²          | Catchment area for runoff nodes                  |
| Water level/elevation   | m a.s.l.     | Reservoir water level (meters above sea level)   |
| Coordinates             | m (UTM)      | Easting/Northing for node locations              |
| Time step duration      | s            | Model time step (typically monthly, in seconds)  |

---

## CSV File Format Examples

**SupplyNode / DemandNode / SinkNode (time-varying rates):**
```csv
Date,Q
2020-01-01,20.2
2020-02-01,33.7
2020-03-01,58.0
...
```
- `Q` is the flow rate in m³/s for each time step.

**RunoffNode (rainfall input):**
```csv
Date,Precipitation
2020-01-01,32.5
2020-02-01,28.1
...
```
- `Precipitation` is in mm for each time step.

**StorageNode (height-volume relationship):**
```csv
h,v
1000,500000
1005,700000
1010,900000
...
```
- `Height` in m a.s.l., `Volume` in m³, `Area` in m².

**StorageNode (evaporation rates):**
```csv
Date,Evaporation
2020-01-01,80.0
2020-02-01,65.0
...
```
- `Evaporation` is in mm per time step.

---

## 1. **SupplyNode**

**Main Function:**  
Represents a water source (e.g., river inflow, intake, external supply) that introduces water into the system.

**Key Parameters:**
- `id` (str, required): Unique identifier for the node.
- `easting`, `northing` (float, required): UTM coordinates for spatial placement.
- `constant_supply_rate` (float, optional): Constant supply rate [m³/s].
- `csv_file` (str, optional): Path to CSV file with time-varying supply data (column: `Q`).
- `start_year`, `start_month` (int, required if using CSV): Start of the time series.
- `num_time_steps` (int, required): Number of simulation steps.

**Parameterization:**
- If both `csv_file` and time parameters are provided, the supply is read from the CSV file.
- If only `constant_supply_rate` is given, the supply is constant for all time steps.
- If neither is provided, supply defaults to zero.

---

## 2. **RunoffNode**

**Main Function:**  
Models distributed surface runoff from a sub-catchment, generating inflow based on rainfall and catchment characteristics.

**Key Parameters:**
- `id` (str, required): Unique identifier.
- `easting`, `northing` (float, required): UTM coordinates.
- `area` (float, required): Catchment area [km²].
- `runoff_coefficient` (float, required): Fraction of rainfall that becomes runoff (0–1).
- `rainfall_csv` (str, required): Path to CSV file with rainfall data (column: `Precipitation` in mm).
- `start_year`, `start_month` (int, required): Start of the rainfall time series.
- `num_time_steps` (int, required): Number of simulation steps.

**Parameterization:**
- Runoff is calculated as and transformed to m^3/s:  
  `Runoff = Rainfall × Area × Runoff_coefficient`
- Rainfall is read from the CSV file for each time step.

---

## 3. **DemandNode**

**Main Function:**  
Represents water users (agriculture, domestic, industrial). Handles both consumptive and non-consumptive demands, and can model delivery/application losses.

**Key Parameters:**
- `id` (str, required): Unique identifier.
- `easting`, `northing` (float, required): UTM coordinates.
- `constant_demand_rate` (float, optional): Constant demand [m³/s].
- `csv_file` (str, optional): Path to CSV file with time-varying demand (column: `Q`).
- `non_consumptive_rate` (float, optional): Portion of demand returned to the system [m³/s].
- `field_efficiency` (float, optional): Fraction of water effectively used at the field (0–1).
- `conveyance_efficiency` (float, optional): Fraction of water delivered to the field (0–1).
- `priority` (int, optional): Demand priority (1=high, 2=low; default=2).
- `start_year`, `start_month` (int, required if using CSV): Start of the demand time series.
- `num_time_steps` (int, required): Number of simulation steps.

**Parameterization:**
- Demand can be constant or time-varying (CSV).
- **Efficiency adjustment:**  
  Gross demand is calculated as:  
  `Gross demand = Net demand / (field_efficiency × conveyance_efficiency)`
- Non-consumptive rate is returned to the system downstream.

---

## 4. **StorageNode**

**Main Function:**  
Models reservoirs or storage facilities. Tracks storage, manages releases, and accounts for evaporation and dead storage.

**Key Parameters:**
- `id` (str, required): Unique identifier.
- `easting`, `northing` (float, required): UTM coordinates.
- `hv_file` (str, required): Path to CSV file with height-volume relationship.
- `evaporation_file` (str, optional): Path to CSV file with monthly evaporation rates [mm].
- `start_year`, `start_month` (int, required if using evaporation): Start of evaporation time series.
- `num_time_steps` (int, required): Number of simulation steps.
- `initial_storage` (float, optional): Initial storage volume [m³].
- `dead_storage` (float, optional): Minimum operational storage [m³].
- `buffer_coefficient` (float, optional): Controls releases in the buffer zone (0–1).

**Parameterization:**
- **Height-volume relationship:**  
  Used to convert between water level and storage volume for water balance and evaporation calculations.
- **Evaporation:**  
  Calculated using surface area (from HV curve) and evaporation rates.
- **Release policy:**  
  Defined by rule-curve parameters (`Vr`, `V1`, `V2`), which set target releases and operational zones:
    - Below `dead_storage`: No release.
    - Buffer zone (`dead_storage` to `V1`): Reduced releases, scaled by `buffer_coefficient`.
    - Conservation zone (`V1` to `V2`): Normal operation, target releases.
    - Above `V2`: Flood control, increased releases.
- **Buffer coefficient:**  
  Determines how much water is released in the buffer zone. Lower values mean more conservative (less) release.

 >**Note:**`Vr`, `V1` and `V2` are defined as decision variables and found during optimization.  

---

## 5. **SinkNode**

**Main Function:**  
Represents endpoints where water leaves the system (e.g., river mouth, environmental flow). Can enforce minimum flow requirements.

**Key Parameters:**
- `id` (str, required): Unique identifier.
- `easting`, `northing` (float, required): UTM coordinates.
- `constant_min_flow` (float, optional): Constant minimum required flow [m³/s].
- `csv_file` (str, optional): Path to CSV file with time-varying minimum flow (column: `Q`).
- `start_year`, `start_month` (int, required if using CSV): Start of the minimum flow time series.
- `num_time_steps` (int, required): Number of simulation steps.

**Parameterization:**
- Minimum flow can be constant or time-varying (CSV).
- If actual flow is below the minimum, the deficit is recorded for reporting and optimization.

---

## 6. **HydroWorks**

**Main Function:**  
Represents distribution or control structures (e.g., diversions, confluences, splitters). Allows splitting and routing flows to multiple downstream nodes.

**Key Parameters:**
- `id` (str, required): Unique identifier.
- `easting`, `northing` (float, required): UTM coordinates.
- **Distribution parameters:**  
  Set via `set_distribution_parameters()`, which defines the fraction of inflow sent to each outflow edge (can be monthly or constant).

**Parameterization:**
- Each outflow edge is assigned a distribution ratio (0–1), and all ratios for a given month must sum to 1.
- Can be used to model operational rules for diversions, bifurcations, or confluences.

---

## Summary Table

| Node Type      | Main Functionality | Key Parameters (see above for details) |
|----------------|--------------------|----------------------------------------|
| SupplyNode     | Water source       | id, easting, northing, constant_supply_rate, csv_file, start_year, start_month, num_time_steps |
| RunoffNode     | Surface runoff     | id, easting, northing, area, runoff_coefficient, rainfall_csv, start_year, start_month, num_time_steps |
| DemandNode     | Water user         | id, easting, northing, constant_demand_rate, csv_file, non_consumptive_rate, field_efficiency, conveyance_efficiency, priority, start_year, start_month, num_time_steps |
| StorageNode    | Reservoir/storage  | id, easting, northing, hv_file, evaporation_file, start_year, start_month, num_time_steps, initial_storage, dead_storage, buffer_coefficient |
| SinkNode       | System outlet      | id, easting, northing, constant_min_flow, csv_file, start_year, start_month, num_time_steps |
| HydroWorks     | Flow distribution  | id, easting, northing, distribution parameters (set via method) |

---