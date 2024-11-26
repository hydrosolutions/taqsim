"""
This module defines the various types of nodes that can exist in a water system simulation.

The module includes a base Node class and several specialized node types such as
SupplyNode, SinkNode, DemandNode, StorageNode, and HydroWorks.
Each node type has its own behavior for handling water inflows and outflows.
"""
import pandas as pd
from scipy.interpolate import interp1d

class Node:
    """
    Base class for all types of nodes in the water system.

    Attributes:
        id (str): A unique identifier for the node.
        inflows (dict): A dictionary of inflow edges, keyed by the source node's id.
        outflows (dict): A dictionary of outflow edges, keyed by the target node's id.
        easting (float): The easting coordinate of the node.
        northing (float): The northing coordinate of the node.
    """

    def __init__(self, id, easting=None, northing=None):
        """
        Initialize a Node object.

        Args:
            id (str): A unique identifier for the node.
            easting (float, optional): The easting coordinate of the node.
            northing (float, optional): The northing coordinate of the node.

        Raises:
            ValueError: If id is empty or coordinates are invalid.
        """
        if not id or not isinstance(id, str):
            raise ValueError(f"Invalid node ID: {id}")
        
        if easting is None or northing is None:
            raise ValueError(f"Missing coordinate value for node {id}: easting={easting}, northing={northing}")
        if not isinstance(easting, (int, float)) or not isinstance(northing, (int, float)):
            raise ValueError(f"Invalid coordinate type for node {id}: easting={easting}, northing={northing}")


        self.id = id
        self.inflow_edges = {}  # Dictionary of inflow edges
        self.outflow_edges = {}  # Dictionary of outflow edges
        self.easting = easting # easting coordinate of the node. Defaults to None.
        self.northing = northing # northing coordinate of the node. Defaults to None.

    def add_inflow(self, edge):
        """
        Add an inflow edge to the node.

        Args:
            edge (Edge): The edge to be added as an inflow.
        """
        self.inflow_edges[edge.source.id] = edge

    def add_outflow(self, edge):
        """
        Add an outflow edge to the node.

        Args:
            edge (Edge): The edge to be added as an outflow.
        """
        self.outflow_edges[edge.target.id] = edge

    def update(self, time_step, dt):
        """
        Update the node's state for the given time step.

        This method should be overridden by subclasses to implement
        specific behavior for each node type.

        Args:
            time_step (int): The current time step of the simulation.
            dt (float): Number of seconds in time step.
        """
        pass

class SupplyNode(Node):
    """
    Represents a water supply source in the system.

    Attributes:
        supply_rates (list): A list of supply rates for each time step.
        default_supply_rate (float): The default supply rate if not specified for a time step.
        supply_history (list): A record of actual supply amounts for each time step.
    """

    def __init__(self, id, supply_rates=None, default_supply_rate=0, easting=None, northing=None,
                 csv_file=None, start_year=None, start_month=None, num_time_steps=None):
        """
        Initialize a SupplyNode object.

        Args:
            id (str): A unique identifier for the node.
            supply_rates (list, optional): A list of supply rates for each time step. Defaults to None.
            default_supply_rate (float, optional): The default supply rate. Defaults to 0.
            easting (float, optional): The easting coordinate of the node. Defaults to None.
            northing (float, optional): The northing coordinate of the node. Defaults to None.
            csv_file (str, optional): Path to CSV file containing supply data. Defaults to None.
            start_year (int, optional): Starting year for CSV data import. Defaults to None.
            start_month (int, optional): Starting month (1-12) for CSV data import. Defaults to None.
            num_time_steps (int, optional): Number of time steps to import from CSV. Defaults to None.
        """
        super().__init__(id, easting, northing)
        self.default_supply_rate = default_supply_rate
        self.supply_history = []

        self.supply_rates = self._initialize_supply_rates(
            id, csv_file, start_year, start_month, num_time_steps, supply_rates
        )

    def _initialize_supply_rates(self, id, csv_file, start_year, start_month, 
                                 num_time_steps, supply_rates):
        """
        Initialize supply rates from either CSV or direct input.
        
        Args:
            id (str): Node identifier for error messages
            csv_file (str): Path to CSV file
            start_year (int): Start year for data
            start_month (int): Start month for data
            num_time_steps (int): Number of time steps
            supply_rates (list): Direct supply rates input
            
        Returns:
            list: Initialized supply rates
        """
        # If all CSV parameters are provided, try to import data
        if all(param is not None for param in [csv_file, start_year, start_month, num_time_steps]):
            try:
                supply = self.import_supply_data(csv_file, start_year, start_month, num_time_steps)
                
                # Check if data is valid
                if not (supply.empty or 
                    supply['Date'].iloc[0] != pd.Timestamp(year=start_year, month=start_month, day=1) or 
                    len(supply['Q']) < num_time_steps):
                    return supply['Q'].tolist()
                
                # Print warning for invalid data
                print(f"Warning: Using default supply rate ({self.default_supply_rate}) for node '{id}' due to insufficient data")
                print(f"Requested period: {start_year}-{start_month:02d} to "
                    f"{pd.Timestamp(year=start_year, month=start_month, day=1) + pd.DateOffset(months=num_time_steps-1):%Y-%m}")
                if not supply.empty:
                    print(f"Available data range: {supply['Date'].min():%Y-%m} to {supply['Date'].max():%Y-%m}")
                return [self.default_supply_rate] * num_time_steps
                
            except Exception as e:
                print(f"Warning: Using default supply rate ({self.default_supply_rate}) for node '{id}' due to error: {str(e)}")
                return [self.default_supply_rate] * num_time_steps
        
        # If no CSV import, use provided rates or initialize empty list
        return supply_rates if supply_rates is not None else []
            
    def import_supply_data(self, csv_file, start_year, start_month, num_time_steps):
        """
        Import supply data from a CSV file for a specified time period.
        
        Args:
            csv_file (str): Path to the CSV file containing supply data
            start_year (int): Starting year for the data
            start_month (int): Starting month (1-12) for the data
            num_time_steps (int): Number of time steps to import
            
        Returns:
            DataFrame: Filtered DataFrame containing the requested data period
        """
        try:
            # Read the CSV file into a pandas DataFrame
            supply = pd.read_csv(csv_file, parse_dates=['Date'])
            
            if 'Date' not in supply.columns or 'Q' not in supply.columns:
                raise ValueError("CSV file must contain 'Date' and 'Q' columns")
        
            # Filter the DataFrame to find the start point
            start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
            end_date = start_date + pd.DateOffset(months=num_time_steps)
            
            return supply[(supply['Date'] >= start_date) & (supply['Date'] < end_date)]
            
        except FileNotFoundError:
            raise ValueError(f"Supply data file not found: {csv_file}")
        except Exception as e:
            raise ValueError(f"Failed to import supply data: {str(e)}")
        
    def get_supply_rate(self, time_step):
        """
        Get the supply rate for a specific time step.

        Args:
            time_step (int): The time step for which to retrieve the supply rate.

        Returns:
            float: The supply rate for the specified time step, or the default rate if not specified.
        """
        if time_step < len(self.supply_rates):
            return self.supply_rates[time_step]
        return self.default_supply_rate

    def update(self, time_step, dt):
        """
        Update the SupplyNode's state for the given time step.

        This method calculates the current supply rate and distributes it among outflow edges.

        Args:
            time_step (int): The current time step of the simulation.
            dt (float): The duration of the time step in seconds.
        """
        try:
            current_supply_rate = self.get_supply_rate(time_step)
            self.supply_history.append(current_supply_rate)

            total_capacity = sum(edge.capacity for edge in self.outflow_edges.values())
            if total_capacity > 0:
                for edge in self.outflow_edges.values():
                    edge_flow = (edge.capacity / total_capacity) * current_supply_rate
                    edge.update(time_step, edge_flow)
            else:
                for edge in self.outflow_edges.values():
                    edge.update(time_step, 0)
        except Exception as e:
            raise ValueError(f"Failed to update supply node {self.id}: {str(e)}")

class SinkNode(Node):
    """
    Represents a point where water exits the system.
    """

    def update(self, time_step, dt):
        """
        Update the SinkNode's state for the given time step.

        This method calculates the total inflow to the sink node.
        Sink nodes remove all incoming water from the system.

        Args:
            time_step (int): The current time step of the simulation.
            dt (float): The duration of the time step in seconds.

        """
        try:
            total_inflow = sum(edge.get_edge_outflow(time_step) for edge in self.inflow_edges.values())
            # Sink nodes remove all incoming water from the system
        except Exception as e:
            raise ValueError(f"Failed to update sink node {self.id}: {str(e)}")

class DemandNode(Node):
    """
    Represents a point of water demand in the system.

    Attributes:
        demand_rates (list): A list of demand rates for each time step.
        satisfied_demand (list): A record of satisfied demand for each time step.
        excess_flow (list): A record of excess flow for each time step.
    """

    def __init__(self, id, demand_rates=None, easting=None, northing=None,
                 csv_file=None, start_year=None, start_month=None, num_time_steps=None, field_efficiency=1):
        """
        Initialize a DemandNode object.

        Args:
            id (str): A unique identifier for the node.
            demand_rates (list or float): Either a list of demand rates for each time step,
                                          or a constant demand rate.
        """
        super().__init__(id, easting, northing)

        # Validate field efficiency
        if not isinstance(field_efficiency, (int, float)):
            raise ValueError("Field efficiency must be a number")
        if field_efficiency <= 0 or field_efficiency > 1:
            raise ValueError("Field efficiency must be between 0 and 1")
        self.field_efficiency = field_efficiency

        if all(param is not None for param in [csv_file, start_year, start_month, num_time_steps]):
            self.demand_rates = self._initialize_demand_rates(
                id, csv_file, start_year, start_month, num_time_steps, demand_rates
            )
        elif isinstance(demand_rates, (int, float)):
            if demand_rates < 0:
                raise ValueError("Demand rate cannot be negative")
            self.demand_rates = [demand_rates/self.field_efficiency]
        elif isinstance(demand_rates, list):
            if not all(isinstance(rate, (int, float)) for rate in demand_rates):
                raise ValueError("All demand rates must be numeric values")
            if any(rate < 0 for rate in demand_rates):
                raise ValueError("Demand rates cannot be negative")
            self.demand_rates = [rate/self.field_efficiency for rate in demand_rates]
        else:
            raise ValueError("demand_rates must be a number or list of numbers or defined by CSV")
            
        self.satisfied_demand = []
        self.excess_flow = []
    
    def _initialize_demand_rates(self, id, csv_file, start_year, start_month, 
                               num_time_steps, demand_rates):
        """
        Initialize demand rates from either CSV or direct input.
        
        Args:
            id (str): Node identifier for error messages
            csv_file (str): Path to CSV file
            start_year (int): Start year for data
            start_month (int): Start month for data
            num_time_steps (int): Number of time steps
            demand_rates (list or float): Direct demand rates input
            
        Returns:
            list: Initialized demand rates
        """
        # If all CSV parameters are provided, try to import data
        if all(param is not None for param in [csv_file, start_year, start_month, num_time_steps]):
            try:
                demand = self.import_demand_data(csv_file, start_year, start_month, num_time_steps)
                
                # Check if data is valid
                if not (demand.empty or 
                    demand['Date'].iloc[0] != pd.Timestamp(year=start_year, month=start_month, day=1) or 
                    len(demand['ETblue [m^3/s]']) < num_time_steps):
                    return [rate/self.field_efficiency for rate in demand['ETblue [m^3/s]'].tolist()]
                
                # Print warning for invalid data
                print(f"Warning: Insufficient data in csv file for node '{id}'")
                print(f"Requested period: {start_year}-{start_month:02d} to "
                    f"{pd.Timestamp(year=start_year, month=start_month, day=1) + pd.DateOffset(months=num_time_steps-1):%Y-%m}")
                if not demand.empty:
                    print(f"Available data range: {demand['Date'].min():%Y-%m} to {demand['Date'].max():%Y-%m}")
            except Exception as e:
                print(f"Warning: Demand from csv could not be used for node '{id}' due to error: {str(e)}")
    
    def import_demand_data(self, csv_file, start_year, start_month, num_time_steps):
        """
        Import demand data from a CSV file for a specified time period.
        
        Args:
            csv_file (str): Path to the CSV file containing demand data
            start_year (int): Starting year for the data
            start_month (int): Starting month (1-12) for the data
            num_time_steps (int): Number of time steps to import
            
        Returns:
            DataFrame: Filtered DataFrame containing the requested data period
        """
        try:
            # Read the CSV file into a pandas DataFrame
            demand = pd.read_csv(csv_file, parse_dates=['Date'])
            
            if 'Date' not in demand.columns or 'ETblue [m^3/s]' not in demand.columns:
                raise ValueError("CSV file must contain 'Date' and 'ETblue [m^3/s]' columns")
        
            # Filter the DataFrame to find the start point
            start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
            end_date = start_date + pd.DateOffset(months=num_time_steps)
            
            return demand[(demand['Date'] >= start_date) & (demand['Date'] < end_date)]
            
        except FileNotFoundError:
            raise ValueError(f"Demand data file not found: {csv_file}")
        except Exception as e:
            raise ValueError(f"Failed to import demand data: {str(e)}")

    def get_demand_rate(self, time_step):
        """
        Get the demand rate for a specific time step.

        Args:
            time_step (int): The time step for which to retrieve the demand rate.

        Returns:
            float: The demand rate for the specified time step, or the last known rate if out of range.
        """
        if time_step < len(self.demand_rates):
            return self.demand_rates[time_step]
        return self.demand_rates[-1]

    def update(self, time_step, dt):
        """
        Update the DemandNode's state for the given time step.

        This method calculates the satisfied demand and excess flow, and distributes
        excess water to outflow edges.

        Args:
            time_step (int): The current time step of the simulation.
            dt (float): The duration of the time step in seconds.
        """
        try:
            total_inflow = sum(edge.get_edge_outflow(time_step) for edge in self.inflow_edges.values())
            current_demand = self.get_demand_rate(time_step)
            satisfied = min(total_inflow, current_demand)
            excess = max(0, total_inflow - current_demand)
            
            self.satisfied_demand.append(satisfied)
            self.excess_flow.append(excess)

            total_outflow_capacity = sum(edge.capacity for edge in self.outflow_edges.values())
            if total_outflow_capacity > 0:
                for edge in self.outflow_edges.values():
                    edge_flow = (edge.capacity / total_outflow_capacity) * excess
                    edge.update(time_step, edge_flow)
            else:
                for edge in self.outflow_edges.values():
                    edge.update(time_step, 0)
        except Exception as e:
            raise ValueError(f"Failed to update demand node {self.id}: {str(e)}")

    def get_satisfied_demand(self, time_step):
        """
        Get the satisfied demand for a specific time step.

        Args:
            time_step (int): The time step for which to retrieve the satisfied demand.

        Returns:
            float: The satisfied demand for the specified time step, or 0 if not available.
        """
        if time_step < len(self.satisfied_demand):
            return self.satisfied_demand[time_step]
        return 0

class StorageNode(Node):  
    """
    Represents a water storage facility in the system.
    Now enhanced with height-volume-area relationships from survey data.

    Attributes:
        id (str): Unique identifier for the node
        capacity (float): Maximum storage capacity [m³]
        storage (list): Record of storage levels for each time step [m³]
        hva_data (dict): Dictionary containing the height-volume-area relationships
        evaporation (list): List of monthly evaporation rates [mm/month]
        evaporation_losses (list): Record of volume lost to evaporation each timestep [m³]
    """

    def __init__(self, id, hva_file, initial_storage=0, easting=None, northing=None, 
                 evaporation_file=None, start_year=None, start_month=None, num_time_steps=None):
        """
        Initialize a StorageNode object.

        Args:
            id (str): Unique identifier for the node
            hva_file (str): Path to CSV file containing height-volume-area relationships
            initial_storage (float, optional): Initial storage volume. Defaults to 0.
            easting (float, optional): Easting coordinate
            northing (float, optional): Northing coordinate
            evaporation_file (str, optional): Path to CSV file containing monthly evaporation rates [mm/month]
            start_year (int, optional): Starting year for evaporation data
            start_month (int, optional): Starting month (1-12) for evaporation data
            num_time_steps (int, optional): Number of time steps to import from evaporation data
        """
        # Call parent class (Node) initialization
        super().__init__(id, easting, northing)
        
        # Initialize StorageNode specific attributes
        self.hva_data = None
        self._level_to_volume = None
        self._volume_to_level = None
        self._level_to_area = None
        self.evaporation_losses = []

        # Load height-volume-area data
        self._load_hva_data(hva_file)
        # Initialize evaporation rates
        self.evaporation_rates = self._initialize_evaporation_rates(
            id, evaporation_file, start_year, start_month, num_time_steps
        )

        # Validate initial storage against capacity
        if initial_storage > self.capacity:
            raise ValueError(f"Initial storage ({initial_storage} m³) exceeds maximum capacity ({self.capacity} m³)")
        
        # Initialize storage attributes
        self.storage = [initial_storage]
        self.spillway_register = []
        self.water_level = [self.get_level_from_volume(initial_storage)]

    def _load_hva_data(self, csv_path):
        """Load and validate height-volume-area relationship data."""
        try:
            # Read and validate CSV
            df = pd.read_csv(csv_path, sep=';')
            
            # Check required columns
            required_cols = ['Height_m', 'Volume_m3', 'Area_m2']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                raise ValueError(f"Missing required columns: {missing}")
            
            # Sort by height and remove duplicates
            df = df.sort_values('Height_m').drop_duplicates(subset=['Height_m'])
            
            # Convert elevations to water levels (depth above ground)
            min_elevation = df['Height_m'].min()
            df['Water_Level_m'] = df['Height_m'] - min_elevation
            
            # Store original elevations for reference
            bottom_elevation = min_elevation
            max_elevation = df['Height_m'].max()

            # Set capacity from maximum volume in survey data
            self.capacity = float(df['Volume_m3'].max())
            
            # Store survey data in dictionary
            self.hva_data = {
                'water_levels': df['Water_Level_m'].values,  # Depth above ground
                'elevations': df['Height_m'].values,         # Original elevations
                'volumes': df['Volume_m3'].values,
                'areas': df['Area_m2'].values,
                'bottom_elevation': bottom_elevation,        # Ground level
                'max_elevation': max_elevation,              # Maximum elevation
                'max_depth': max_elevation - min_elevation   # Maximum water depth
            }
            
            # Initialize interpolation functions
            self._initialize_interpolators()
            
        except Exception as e:
            raise ValueError(f"Error loading HVA data from CSV file: {str(e)}")

    def _initialize_evaporation_rates(self, id, evaporation_file, start_year, start_month, num_time_steps):
        """
        Initialize evaporation rates from CSV file.
        
        Args:
            id (str): Node identifier for error messages
            evaporation_file (str): Path to evaporation data CSV
            start_year (int): Start year for data
            start_month (int): Start month for data
            num_time_steps (int): Number of time steps
            
        Returns:
            list: Initialized evaporation rates or None if not configured
        """
        # If no evaporation file provided, return None
        if evaporation_file is None:
            return None

        # If all parameters are provided, try to import data
        if all(param is not None for param in [evaporation_file, start_year, start_month, num_time_steps]):
            try:
                evap_data = self.import_evaporation_data(evaporation_file, start_year, start_month, num_time_steps)
                
                # Check if data is valid
                if not (evap_data.empty or 
                    evap_data['Date'].iloc[0] != pd.Timestamp(year=start_year, month=start_month, day=1) or 
                    len(evap_data['Evaporation']) < num_time_steps):
                    return evap_data['Evaporation'].tolist()
                
                # Print warning for invalid data
                print(f"Warning: No evaporation rates will be applied for node '{id}' due to insufficient data")
                print(f"Requested period: {start_year}-{start_month:02d} to "
                    f"{pd.Timestamp(year=start_year, month=start_month, day=1) + pd.DateOffset(months=num_time_steps-1):%Y-%m}")
                if not evap_data.empty:
                    print(f"Available data range: {evap_data['Date'].min():%Y-%m} to {evap_data['Date'].max():%Y-%m}")
                return None
                
            except Exception as e:
                print(f"Warning: Failed to load evaporation data for node '{id}': {str(e)}")
                return None
        
        return None

    def import_evaporation_data(self, evaporation_file, start_year, start_month, num_time_steps):
        """
        Import evaporation data from a CSV file for a specified time period.
        
        Args:
            evaporation_file (str): Path to the CSV file containing evaporation data
            start_year (int): Starting year for the data
            start_month (int): Starting month (1-12) for the data
            num_time_steps (int): Number of time steps to import
            
        Returns:
            DataFrame: Filtered DataFrame containing the requested data period
        """
        try:
            # Read the CSV file into a pandas DataFrame
            evap_df = pd.read_csv(evaporation_file, parse_dates=['Date'])
            
            if 'Date' not in evap_df.columns or 'Evaporation' not in evap_df.columns:
                raise ValueError("CSV file must contain 'Date' and 'Evaporation' columns")
        
            # Filter the DataFrame to find the start point
            start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
            end_date = start_date + pd.DateOffset(months=num_time_steps)
            
            return evap_df[(evap_df['Date'] >= start_date) & (evap_df['Date'] < end_date)]
            
        except FileNotFoundError:
            raise ValueError(f"Evaporation data file not found: {evaporation_file}")
        except Exception as e:
            raise ValueError(f"Failed to import evaporation data: {str(e)}")

    def _initialize_interpolators(self):
        """Initialize interpolation functions for height-volume-area relationships."""
        try:
            # Create height to volume interpolator
            self._level_to_volume = interp1d(
                self.hva_data['water_levels'],
                self.hva_data['volumes'],
                kind='linear',
                bounds_error=False,  # Allow extrapolation
                fill_value=(self.hva_data['volumes'][0], self.hva_data['volumes'][-1])
            )
            
            # Create volume to height interpolator
            self._volume_to_level = interp1d(
                self.hva_data['volumes'],
                self.hva_data['water_levels'],
                kind='linear',
                bounds_error=False,
                fill_value=(self.hva_data['water_levels'][0], self.hva_data['water_levels'][-1])
            )
            
            # Create height to area interpolator
            self._level_to_area = interp1d(
                self.hva_data['water_levels'],
                self.hva_data['areas'],
                kind='linear',
                bounds_error=False,
                fill_value=(self.hva_data['areas'][0], self.hva_data['areas'][-1])
            )
            
        except Exception as e:
            raise Exception(f"Error creating interpolation functions: {str(e)}")

    def get_volume_from_level(self, water_level):
        """
        Get storage volume for a given water level.
        
        Args:
            water_level (float): Water level above ground [m]
            
        Returns:
            float: Corresponding storage volume [m³]
        """
        if not self.hva_data:
            print(f'{self.id} volume can not be determined from water level: Height-Volume relation is missing!')
            return

        if self._level_to_volume is None:
            raise ValueError("No level-volume relationship available")
        return float(self._level_to_volume(water_level))
        
    def get_level_from_volume(self, volume):
        """
        Get water level for a given storage volume.
        
        Args:
            volume (float): Storage volume [m³]
            
        Returns:
            float: Corresponding water level above ground [m]
        """
        if not self.hva_data:
            print(f'{self.id} water level can not be determined from volume: Height-Volume relation is missing!')
            return

        if self._volume_to_level is None:
            raise ValueError("No volume-level relationship available")
        return float(self._volume_to_level(volume))

    def get_area_from_level(self, water_level):
        """
        Get surface area for a given water level.
        
        Args:
            water_level (float): Water level above ground [m]
            
        Returns:
            float: Corresponding surface area [m²]
        """
        if not self.hva_data:
            print(f'{self.id} water surface area can not be determined from water level: Height-Area relation is missing!')
            return
        
        if self._level_to_area is None:
            raise ValueError("No level-area relationship available")
        return float(self._level_to_area(water_level))

    def get_elevation_from_level(self, water_level):
        """Convert water level to elevation."""
        if not self.hva_data:
            print(f'{self.id}: Height-Volume-Area relation is missing!')
            return
        return self.hva_data['bottom_elevation'] + water_level

    def get_level_from_elevation(self, elevation):
        """Convert elevation to water level."""
        if not self.hva_data:
            print(f'{self.id}: Height-Volume-Area relation is missing!')
            return
        return elevation - self.hva_data['bottom_elevation']
    
    def get_reservoir_info(self):
        """
        Get basic reservoir information.
        
        Returns:
            dict: Dictionary containing reservoir characteristics
        """
        if not self.hva_data:
            print(f'{self.id}: Height-Volume-Area relation is missing!')
            return
        return {
            'bottom_elevation': self.hva_data['bottom_elevation'],
            'max_elevation': self.hva_data['max_elevation'],
            'max_depth': self.hva_data['max_depth'],
            'max_volume': self.capacity,
            'max_area': float(self._level_to_area(self.hva_data['max_depth']))
        }

    def get_interpolation_ranges(self):
        """
        Get the valid ranges for interpolation.
        
        Returns:
            dict: Dictionary containing min/max values for height, volume, and area
        """
        if not self.hva_data:
            return None
            
        return {
            'elevation_range': {
                'min': float(self.hva_data['elevations'][0]),
                'max': float(self.hva_data['elevations'][-1])
            },
            'water_level_range': {
                'min': float(self.hva_data['water_levels'][0]),
                'max': float(self.hva_data['water_levels'][-1])
            },
            'volume_range': {
                'min': float(self.hva_data['volumes'][0]),
                'max': float(self.hva_data['volumes'][-1])
            },
            'area_range': {
                'min': float(self.hva_data['areas'][0]),
                'max': float(self.hva_data['areas'][-1])
            }
        }
    
    def get_evaporation_loss(self, time_step):
        """
        Get the evaporation loss for a specific time step.

        Args:
            time_step (int): The time step for which to retrieve the evaporation loss.

        Returns:
            float: The evaporation loss in m³ for the specified time step, or 0 if not available.
        """
        if time_step < len(self.evaporation_losses):
            return self.evaporation_losses[time_step]
        return 0

    def update(self, time_step, dt):
        """
        Update the StorageNode's state for the given time step.

        This method calculates the new storage level based on inflows, outflows,
        and evaporation losses, and distributes available water to outflow edges.

        Args:
            time_step (int): The current time step of the simulation.
            dt (float): The length of the time step in seconds.
        """
        try:
            inflow = sum(edge.get_edge_outflow(time_step) for edge in self.inflow_edges.values())
            previous_storage = self.storage[-1]
            
            # Convert flow rates (m³/s) to volumes (m³) for the time step
            inflow_volume = inflow * dt
            
            # Calculate evaporation loss
            previous_water_level = self.get_level_from_volume(previous_storage)
            new_water_level = min((previous_water_level - (self.evaporation_rates[time_step] / 1000)),self.hva_data['bottom_elevation'] )  # Convert mm to m

            evap_loss = previous_storage-self.get_volume_from_level(new_water_level)
            self.evaporation_losses.append(evap_loss)
            
             # Calculate available water after evaporation
            available_water = previous_storage + inflow_volume -evap_loss
            available_water = max(0, available_water)  # Ensure non-negative storage
            
            # Calculate total requested outflow
            requested_outflow = sum(edge.capacity for edge in self.outflow_edges.values())
            
            # Convert requested outflow to volume
            requested_outflow_volume = requested_outflow * dt
            
            # Limit actual outflow to available water
            actual_outflow_volume = min(available_water, requested_outflow_volume)
            
            # Distribute actual outflow among edges proportionally
            if requested_outflow_volume > 0:
                for edge in self.outflow_edges.values():
                    edge_flow_volume = (edge.capacity / requested_outflow) * actual_outflow_volume
                    edge_flow_rate = edge_flow_volume / dt
                    edge.update(time_step, edge_flow_rate)
            else:
                for edge in self.outflow_edges.values():
                    edge.update(time_step, 0)
            
            # Calculate new storage
            new_storage = available_water - actual_outflow_volume

            # Check if new storage exceeds maximum and handle spillway
            if new_storage > self.capacity:
                excess_volume = new_storage - self.capacity
                new_storage = self.capacity
            else:
                excess_volume = 0
            # Log the spill event in the storage node's spillway register
            self.spillway_register.append(excess_volume)
            
            self.storage.append(new_storage)
            if self.hva_data:
                self.water_level.append(self.get_level_from_volume(new_storage))
        except Exception as e:
            # Log the error and attempt to maintain last known state
            print(f"Error updating storage node {self.id}: {str(e)}")

    def get_storage(self, time_step):
        """
        Get the storage level for a specific time step.

        Args:
            time_step (int): The time step for which to retrieve the storage level.

        Returns:
            float: The storage level in cubic meters for the specified time step, or the last known storage level if out of range.
        """
        if time_step < len(self.storage):
            return self.storage[time_step]
        return self.storage[-1]

class HydroWorks(Node):
    """
    Represents a point where water can be redistributed, combining the functionality
    of diversion and confluence points.
    """

    def update(self, time_step, dt):
        """
        Update the HydroWorks node's state for the given time step.

        This method calculates the total inflow and distributes it among outflow edges
        based on their capacities.

        Args:
            time_step (int): The current time step of the simulation.
            dt (float): The duration of the time step in seconds.
        """
        try:
            total_inflow = sum(edge.get_edge_outflow(time_step) for edge in self.inflow_edges.values())
            total_outflow_capacity = sum(edge.capacity for edge in self.outflow_edges.values())

            if total_outflow_capacity > 0:
                for edge in self.outflow_edges.values():
                    # Distribute water proportionally based on edge capacity
                    edge_flow = (edge.capacity / total_outflow_capacity) * total_inflow
                    edge.update(time_step, edge_flow)
            else:
                # If there's no outflow capacity, set all outflows to 0
                for edge in self.outflow_edges.values():
                    edge.update(time_step, 0)
        except Exception as e:
            raise ValueError(f"Failed to update hydroworks node {self.id}: {str(e)}")
