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

def run_sample_tests():
    # Test simple system
    simple_system = create_simple_system()
    print("Simple System Test:")
    num_time_steps = 24
    simple_system.simulate(num_time_steps)
    save_water_balance_to_csv(simple_system, "simple_system_balance.csv")

    print("\n" + "="*50 + "\n")

    # Test complex system
    complex_system = create_complex_system()
    print("Complex System Test:")
    num_time_steps = 36
    complex_system.simulate(num_time_steps)
    save_water_balance_to_csv(complex_system, "complex_system_balance.csv")

    print("\n" + "="*50 + "\n")

    # Test drought system
    drought_system = create_drought_system()
    print("Drought System Test:")
    num_time_steps = 120
    drought_system.simulate(num_time_steps)
    save_water_balance_to_csv(drought_system, "drought_system_balance.csv")

# Run the sample tests
if __name__ == "__main__":
    run_sample_tests()