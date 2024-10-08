class WaterSystem:
    """
    Represents a water system as a directed graph and provides methods for simulation and visualization.

    Attributes:
        graph (nx.DiGraph): A NetworkX directed graph representing the water system.
        time_steps (int): The number of time steps in the most recent simulation.
        flow_data (dict): A dictionary to store flow data for each node across time steps.
    """

    def __init__(self):
        """
        Initialize a new WaterSystem instance.
        """
        self.graph = nx.DiGraph()
        self.time_steps = 0
        self.flow_data = {}

    def add_node(self, node):
        """
        Add a node to the water system.

        Args:
            node (Node): The node to be added to the system.

        This method adds the node to the graph and stores its type as an attribute.
        """
        node_type = type(node).__name__
        if node_type == 'SupplyNode':
            self.graph.add_node(node.id, node=node, node_type=node_type, supply_rates=node.supply_rates)
        else:
            self.graph.add_node(node.id, node=node, node_type=node_type)

    def add_edge(self, edge):
        """
        Add an edge to the water system.

        Args:
            edge (Edge): The edge to be added to the system.

        This method adds the edge to the graph, connecting its source and target nodes.
        """
        self.graph.add_edge(edge.source.id, edge.target.id, edge=edge)

    def simulate(self, time_steps):
        """
        Run a simulation of the water system for a specified number of time steps.

        Args:
            time_steps (int): The number of time steps to simulate.

        This method updates each node and edge in the system for each time step and collects flow data.
        """
        self.time_steps = time_steps
        self.flow_data = {node: [] for node in self.graph.nodes()}
        
        for t in range(time_steps):
            for node_id in self.graph.nodes():
                node = self.graph.nodes[node_id]['node']
                node.update(t)
                if isinstance(node, SupplyNode):
                    self.flow_data[node_id].append(node.get_outflow(t))
            for _, _, edge_data in self.graph.edges(data=True):
                edge_data['edge'].update(t)

    def print_flow_data(self):
        """
        Print the flow data for all supply nodes in the system.
        """
        for node_id, flows in self.flow_data.items():
            node = self.graph.nodes[node_id]['node']
            if isinstance(node, SupplyNode):
                print(f"Flow data for SupplyNode {node_id}:")
                for t, flow in enumerate(flows):
                    print(f"  Time step {t}: {flow:.2f}")
                print()

    def visualize(self):
        """
        Visualize the water system using matplotlib.

        This method creates a multipartite layout of the system, with nodes color-coded by type
        and edges labeled with their final flow values.
        """
        # Determine node layers for multipartite layout
        layers = {'SupplyNode': 0, 'StorageNode': 1, 'DemandNode': 2}
        colors = []
        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data['node_type']
            if node_type not in layers:
                layers[node_type] = 1  # Default to middle layer for unknown types
            
            if node_type == 'SupplyNode':
                colors.append('skyblue')
            elif node_type == 'StorageNode':
                colors.append('lightgreen')
            elif node_type == 'DemandNode':
                colors.append('lightcoral')
            else:
                colors.append('lightgray')

        # Create multipartite layout
        pos = nx.multipartite_layout(self.graph, subset_key='node_type')

        plt.figure(figsize=(12, 8))
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color=colors, node_size=500)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', arrows=True)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=10)

        # Draw edge labels (flow values)
        edge_labels = {}
        for (u, v, data) in self.graph.edges(data=True):
            if data['edge'].flow:
                edge_labels[(u, v)] = f"{data['edge'].flow[-1]:.2f}"
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)

        plt.title(f"Water System - Time Step: {self.time_steps}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()