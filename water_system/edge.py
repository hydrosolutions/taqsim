"""
This module defines the Edge class, which represents a connection between two nodes in a water system.

The Edge class is responsible for managing the flow of water between nodes, respecting capacity
constraints and updating flow values at each time step of the simulation.
"""

class Edge:
    """
    Represents a connection (e.g., river, canal) between two nodes in a water system.

    The Edge class manages the flow of water between a source node and a target node,
    respecting the capacity constraints of the connection.

    Attributes:
        source (Node): The node from which water flows.
        target (Node): The node to which water flows.
        capacity (float): The maximum amount of water that can flow through this edge per time step.
        flow (list of float): The amount of water flowing through this edge at each time step.

    """

    def __init__(self, source, target, capacity):
        """
        Initialize an Edge object.

        Args:
            source (Node): The source node of the edge.
            target (Node): The target node of the edge.
            capacity (float): The maximum flow capacity of the edge.

        The edge is automatically added to the source node's outflows and the target node's inflows.
        """
        self.source = source
        self.target = target
        self.capacity = capacity
        self.flow = [0]  # Initialize with zero flow
        self.source.add_outflow(self)
        self.target.add_inflow(self)

    def update(self, time_step):
        """
        Update the flow through this edge for the given time step.

        This method calculates and appends the flow value for the current time step
        based on the source node's type and the edge's capacity.

        Args:
            time_step (int): The current time step of the simulation.

        If the source is a SupplyNode, the flow is set to the minimum of the supply rate
        and the edge's capacity. For other node types, the flow is calculated based on
        the inflow to the source node, limited by the edge's capacity.
        """
        if time_step >= len(self.flow):
            if hasattr(self.source, 'supply_rate'):
                # This is a SupplyNode
                self.flow.append(min(self.source.supply_rate, self.capacity))
            else:
                inflow = sum(edge.flow[-1] for edge in self.source.inflows.values())
                outflow = min(inflow, self.capacity)
                self.flow.append(outflow)
        
    def get_flow(self, time_step):
        """
        Get the flow value for a specific time step.

        Args:
            time_step (int): The time step for which to retrieve the flow value.

        Returns:
            float: The flow value for the specified time step.

        If the requested time step is beyond the current simulation time,
        the method returns the last known flow value.
        """
        if time_step < len(self.flow):
            return self.flow[time_step]
        return self.flow[-1]  # Return the last known flow if time_step is out of range