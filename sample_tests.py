import csv
import networkx as nx 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from water_system import WaterSystem, SupplyNode, StorageNode, DemandNode, SinkNode, Edge

def create_simple_system():
    """
    Creates a simple water system with one supply, one storage, two demands, and one sink.
    """
    system = WaterSystem()
    
    supply = SupplyNode("MainSupply", default_supply_rate=100)
    reservoir = StorageNode("MainReservoir", capacity=500)
    agriculture = DemandNode("Agriculture", demand_rate=60)
    urban = DemandNode("Urban", demand_rate=30)
    sink = SinkNode("Sink")
    
    system.add_node(supply)
    system.add_node(reservoir)
    system.add_node(agriculture)
    system.add_node(urban)
    system.add_node(sink)
    
    system.add_edge(Edge(supply, reservoir, capacity=120))
    system.add_edge(Edge(reservoir, agriculture, capacity=70))
    system.add_edge(Edge(reservoir, urban, capacity=40))
    system.add_edge(Edge(agriculture, sink, capacity=100))
    system.add_edge(Edge(urban, sink, capacity=50))
    
    return system

def create_complex_system():
    """
    Creates a more complex water system with multiple supplies, storages, and demands.
    """
    system = WaterSystem()
    
    supply1 = SupplyNode("MountainSupply", default_supply_rate=150)
    supply2 = SupplyNode("RiverSupply", default_supply_rate=100)
    reservoir1 = StorageNode("MountainReservoir", capacity=1000)
    reservoir2 = StorageNode("ValleyReservoir", capacity=800)
    agriculture1 = DemandNode("Farmland1", demand_rate=80)
    agriculture2 = DemandNode("Farmland2", demand_rate=70)
    urban1 = DemandNode("City1", demand_rate=50)
    urban2 = DemandNode("City2", demand_rate=40)
    industry = DemandNode("IndustrialPark", demand_rate=30)
    sink = SinkNode("RiverMouth")
    
    nodes = [supply1, supply2, reservoir1, reservoir2, agriculture1, agriculture2, 
             urban1, urban2, industry, sink]
    for node in nodes:
        system.add_node(node)
    
    system.add_edge(Edge(supply1, reservoir1, capacity=160))
    system.add_edge(Edge(supply2, reservoir2, capacity=120))
    system.add_edge(Edge(reservoir1, agriculture1, capacity=90))
    system.add_edge(Edge(reservoir1, urban1, capacity=60))
    system.add_edge(Edge(reservoir2, agriculture2, capacity=80))
    system.add_edge(Edge(reservoir2, urban2, capacity=50))
    system.add_edge(Edge(reservoir2, industry, capacity=40))
    system.add_edge(Edge(agriculture1, sink, capacity=100))
    system.add_edge(Edge(agriculture2, sink, capacity=100))
    system.add_edge(Edge(urban1, sink, capacity=70))
    system.add_edge(Edge(urban2, sink, capacity=60))
    system.add_edge(Edge(industry, sink, capacity=50))
    
    return system

def create_drought_system():
    """
    Creates a system to test drought conditions with variable supply.
    """
    system = WaterSystem()
    
    # Alternating between normal (100) and drought (20) conditions
    variable_supply = [100 if i % 2 == 0 else 20 for i in range(50)]
    supply = SupplyNode("VariableSupply", supply_rates=variable_supply)
    reservoir = StorageNode("EmergencyReservoir", capacity=500)
    city = DemandNode("DroughtCity", demand_rate=80)
    agriculture = DemandNode("DroughtAgriculture", demand_rate=40)
    sink = SinkNode("Sink")
    
    system.add_node(supply)
    system.add_node(reservoir)
    system.add_node(city)
    system.add_node(agriculture)
    system.add_node(sink)
    
    system.add_edge(Edge(supply, reservoir, capacity=150))
    system.add_edge(Edge(reservoir, city, capacity=90))
    system.add_edge(Edge(reservoir, agriculture, capacity=50))
    system.add_edge(Edge(city, sink, capacity=100))
    system.add_edge(Edge(agriculture, sink, capacity=60))
    
    return system

def save_water_balance_to_csv(water_system, filename):
    """
    Save the water balance table of a water system to a CSV file.
    
    Args:
    water_system (WaterSystem): The water system to save the balance for.
    filename (str): The name of the CSV file to save to.
    """
    balance_table = water_system.get_water_balance_table()
    balance_table.to_csv(filename, index=False)
    print(f"Water balance table saved to {filename}")

def plot_water_system(water_system, filename):
    """
    Create and save a single plot for the entire water system.
    
    Args:
    water_system (WaterSystem): The water system to plot.
    filename (str): The name of the PNG file to save to.
    """
    balance_table = water_system.get_water_balance_table()
    
    plt.figure(figsize=(12, 8))
    plt.title('Water System Simulation Results')
    
    colors = list(mcolors.TABLEAU_COLORS.values())  # Use predefined Tableau colors
    line_styles = ['-', '--', ':', '-.']
    
    for i, (node_id, node_data) in enumerate(water_system.graph.nodes(data=True)):
        node = node_data['node']
        color = colors[i % len(colors)]
        
        plt.plot(balance_table['TimeStep'], balance_table[f'{node_id}_Inflow'], 
                 color=color, linestyle=line_styles[0], label=f'{node_id} Inflow')
        plt.plot(balance_table['TimeStep'], balance_table[f'{node_id}_Outflow'], 
                 color=color, linestyle=line_styles[1], label=f'{node_id} Outflow')
        
        if isinstance(node, StorageNode):
            plt.plot(balance_table['TimeStep'], balance_table[f'{node_id}_Storage'], 
                     color=color, linestyle=line_styles[2], label=f'{node_id} Storage')
        elif isinstance(node, DemandNode):
            plt.plot(balance_table['TimeStep'], balance_table[f'{node_id}_SatisfiedDemand'], 
                     color=color, linestyle=line_styles[2], label=f'{node_id} Satisfied Demand')
            plt.axhline(y=node.demand_rate, color=color, linestyle=line_styles[3], 
                        label=f'{node_id} Demand')
        elif isinstance(node, SupplyNode):
            plt.plot(balance_table['TimeStep'], balance_table[f'{node_id}_SupplyRate'], 
                     color=color, linestyle=line_styles[2], label=f'{node_id} Supply Rate')
    
    plt.xlabel('Time Step')
    plt.ylabel('Water Units')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Water system plot saved to {filename}")
    
def plot_network_layout(water_system, filename):
    """
    Create and save a network layout plot for the water system.
    
    Args:
    water_system (WaterSystem): The water system to plot.
    filename (str): The name of the PNG file to save to.
    """
    G = water_system.graph
    pos = nx.spring_layout(G, k=0.9, iterations=50)
    
    plt.figure(figsize=(12, 8))
    plt.title('Water System Network Layout')
    
    node_colors = {
        SupplyNode: 'skyblue',
        StorageNode: 'lightgreen',
        DemandNode: 'salmon',
        SinkNode: 'lightgray'
    }
    
    for node_type, color in node_colors.items():
        node_list = [node for node, data in G.nodes(data=True) if isinstance(data['node'], node_type)]
        nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color=color, node_size=300, alpha=0.8)
    
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
    
    labels = {node: f"{data['node'].__class__.__name__}\n{node}" for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    edge_labels = {(u, v): f'{d["edge"].capacity}' for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Network layout plot saved to {filename}")

def run_sample_tests():
    # Test simple system
    simple_system = create_simple_system()
    print("Simple System Test:")
    num_time_steps = 12
    simple_system.simulate(num_time_steps)
    save_water_balance_to_csv(simple_system, "simple_system_balance.csv")
    plot_water_system(simple_system, "simple_system_plot.png")
    plot_network_layout(simple_system, "simple_system_network.png")


    print("\n" + "="*50 + "\n")

    # Test complex system
    complex_system = create_complex_system()
    print("Complex System Test:")
    num_time_steps = 36
    complex_system.simulate(num_time_steps)
    save_water_balance_to_csv(complex_system, "complex_system_balance.csv")
    plot_water_system(complex_system, "complex_system_plot.png")
    plot_network_layout(complex_system, "complex_system_network.png")


    print("\n" + "="*50 + "\n")

    # Test drought system
    drought_system = create_drought_system()
    print("Drought System Test:")
    num_time_steps = 120
    drought_system.simulate(num_time_steps)
    save_water_balance_to_csv(drought_system, "drought_system_balance.csv")
    plot_water_system(drought_system, "drought_system_plot.png")
    plot_network_layout(drought_system, "drought_system_network.png")


# Run the sample tests
if __name__ == "__main__":
    run_sample_tests()