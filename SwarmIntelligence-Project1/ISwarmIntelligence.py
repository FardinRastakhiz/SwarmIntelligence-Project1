from abc import ABC 
from abc import abstractmethod

class SwarmIntelligence(ABC):
    def __init__(self, target_function, input_ranges):
        self.function = target_function
        self.input_ranges = input_ranges
        self.positions : np.ndarray = self.generate_swarm_positions()
        self.velocities : np.ndarray = self.generate_swarm_velocities()
        
class PSO(SwarmIntelligence):
    pass