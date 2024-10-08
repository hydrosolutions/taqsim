from .structure import SupplyNode

class Edge:
    def __init__(self, source, target, capacity):
        self.source = source
        self.target = target
        self.capacity = capacity
        self.flow = []
        self.source.add_outflow(self)
        self.target.add_inflow(self)

    def update(self, time_step, flow=None):
        if flow is not None:
            self.flow.append(min(flow, self.capacity))
        elif isinstance(self.source, SupplyNode):
            supply_rate = self.source.get_supply_rate(time_step)
            self.flow.append(min(supply_rate, self.capacity))
        else:
            # This case should not occur with the new node update methods
            self.flow.append(0)

    def get_flow(self, time_step):
        if time_step < len(self.flow):
            return self.flow[time_step]
        return 0