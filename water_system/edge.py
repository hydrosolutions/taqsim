from .structure import Node

class Edge:
    """
    Represents a connection (e.g., river, canal) between two nodes in a water system.

    The Edge class manages the flow of water between a source node and a target node,
    respecting the capacity constraints of the connection and the specific behaviors
    of different node types.

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
        based on the source node's type and the edge's capacity. It handles SupplyNodes,
        StorageNodes, and other node types differently to ensure proper water balance.

        Args:
            time_step (int): The current time step of the simulation.
        """
        if hasattr(self.source, 'supply_rate'):
            # This is a SupplyNode
            self.flow.append(min(self.source.supply_rate, self.capacity))
        elif not hasattr(self.source, 'storage'):
            # For non-supply and non-storage nodes, calculate flow based on inflows
            if time_step >= len(self.flow):
                inflow = sum(edge.flow[-1] for edge in self.source.inflows.values())
                outflow = min(inflow, self.capacity)
                self.flow.append(outflow)
        # StorageNodes will update the flow themselves in their own update method

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