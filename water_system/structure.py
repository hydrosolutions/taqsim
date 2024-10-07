class Node:
    def __init__(self, id):
        self.id = id
        self.inflows = {}  # Dictionary of inflow edges
        self.outflows = {}  # Dictionary of outflow edges

    def add_inflow(self, edge):
        self.inflows[edge.source.id] = edge

    def add_outflow(self, edge):
        self.outflows[edge.target.id] = edge

    def update(self, time_step):
        pass

class SupplyNode(Node):
    def __init__(self, id, supply_rate):
        super().__init__(id)
        self.supply_rate = supply_rate

    def update(self, time_step):
        for edge in self.outflows.values():
            edge.update(time_step)

class SinkNode(Node):
    def update(self, time_step):
        total_inflow = sum(edge.get_flow(time_step) for edge in self.inflows.values())
        # Sink nodes remove all incoming water from the system

class DemandNode(Node):
    def __init__(self, id, demand_rate):
        super().__init__(id)
        self.demand_rate = demand_rate

    def update(self, time_step):
        total_inflow = sum(edge.get_flow(time_step) for edge in self.inflows.values())
        satisfied_demand = min(total_inflow, self.demand_rate)
        # Remaining water continues downstream

class StorageNode(Node):
    def __init__(self, id, capacity):
        super().__init__(id)
        self.capacity = capacity
        self.storage = [0]

    def update(self, time_step):
        inflow = sum(edge.get_flow(time_step) for edge in self.inflows.values())
        previous_storage = self.storage[-1]
        available_water = previous_storage + inflow
        
        for edge in self.outflows.values():
            edge.update(time_step)
        
        outflow = sum(edge.get_flow(time_step) for edge in self.outflows.values())
        new_storage = min(available_water - outflow, self.capacity)
        self.storage.append(max(new_storage, 0))

class DiversionNode(Node):
    def update(self, time_step):
        total_inflow = sum(edge.get_flow(time_step) for edge in self.inflows.values())
        # Implement diversion logic here
        for edge in self.outflows.values():
            edge.update(time_step)

class ConfluenceNode(Node):
    def update(self, time_step):
        total_inflow = sum(edge.get_flow(time_step) for edge in self.inflows.values())
        # Distribute total inflow among outflow edges
        for edge in self.outflows.values():
            edge.update(time_step)