import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod
from abc import ABC

class SwarmIntelligence(ABC):
    def __init__(self, target_function):
        self.function = target_function

class PSO(SwarmIntelligence):
    pass

class SimpleRealPSO(PSO):
    
    def __init__(self, target_function, intertia_weight_function, input_ranges: np.ndarray, swarm_size: int=100, c1: int=0.3, c2: int=0.4, iteration_count: int=100):
        super().__init__(target_function)
        self.w = intertia_weight_function
        self.N = swarm_size
        self.d = input_ranges.shape[0]
        self.input_ranges = input_ranges
        self.c1 = c1
        self.c2 = c2
        self.iteration_count = iteration_count
        print(self.function)

        self.positions : np.ndarray = self.generate_swarm_positions()
        self.velocities : np.ndarray = self.generate_swarm_velocities()
        self.fitnesses : np.ndarray = self.evaluate_fitnesses()

        self.pbests = self.positions.copy()
        self.pbest_fitnesses = self.fitnesses.copy()
        self.gbests : np.ndarray = self.take_gbests()

    def run(self):
        for i in range(self.iteration_count):
            self.update_velocities(i)
            self.update_positions()
            self.fitnesses = self.evaluate_fitnesses()
            self.update_pbests()
            self.gbests = self.take_gbests()
            print(self.fitnesses[0])

    def generate_swarm_positions(self):
        positions = np.random.rand(self.N, self.d)
        lower_bound = self.input_ranges[:, 0][None, :]
        upper_bound = self.input_ranges[:, 1][None, :]
        positions = lower_bound + (upper_bound - lower_bound) * positions
        return positions

    def generate_swarm_velocities(self):
        velocities = np.random.rand(self.N, self.d)
        lower_bound = self.input_ranges[:, 0][None, :]
        upper_bound = self.input_ranges[:, 1][None, :]
        velocities = ((upper_bound - lower_bound) / 2.0) * velocities
        return velocities

    def evaluate_fitnesses(self):
        return self.function(self.positions.T)

    def update_pbests(self):
        self.pbest_fitnesses = self.function(self.pbests.T)
        changes = self.pbest_fitnesses > self.fitnesses
        self.pbests[changes] = self.positions[changes]

    def take_gbests(self):
        return self.pbests[np.argmax(self.pbest_fitnesses)]

    def update_velocities(self, t: int):
        inertia = self.w(t) * self.velocities 
        cognitive = self.c1 * np.random.rand(self.N)[:, None] * (self.pbests - self.positions)
        social = self.c2 * np.random.rand(self.N)[:, None] * (self.gbests - self.positions)
        self.velocities = inertia + cognitive + social

    def update_positions(self):
        self.positions = self.positions + self.velocities

def main():
    iteration_count = 1000
    target_function = lambda X: 3 * X[0] ** 2 - 2 * X[0] * X[1] + 3 * X[1] ** 2 - X[0] - X[1]
    weight_function = lambda t:  0.9 - (0.7 / iteration_count) * t
    input_ranges = np.array([[-5, 5], [-5, 5]])
    spso = SimpleRealPSO(target_function, weight_function, input_ranges, iteration_count)
    spso.run()

if __name__ == "__main__":
    main()
