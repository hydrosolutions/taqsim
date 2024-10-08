class Edge:
    def __init__(self, source, target, capacity):
        self.source = source
        self.target = target
        self.capacity = capacity
        self.flow = [0]  # Initialize with zero flow
        self.source.add_outflow(self)
        self.target.add_inflow(self)

    def update(self, time_step, flow=None):
        if time_step >= len(self.flow):
            if flow is not None:
                self.flow.append(min(flow, self.capacity))
            elif hasattr(self.source, 'get_supply_rate'):
                self.flow.append(min(self.source.get_supply_rate(time_step), self.capacity))
            else:
                inflow = sum(edge.get_flow(time_step) for edge in self.source.inflow_edges.values())
                outflow = min(inflow, self.capacity)
                self.flow.append(outflow)

    def get_flow(self, time_step):
        if time_step < len(self.flow):
            return self.flow[time_step]
        return self.flow[-1]  # Return the last known flow if time_step is out of range