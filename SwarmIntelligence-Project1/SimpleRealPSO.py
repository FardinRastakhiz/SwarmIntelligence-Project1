from ISwarmIntelligence import PSO
import numpy as np
from abc import ABC
from abc import abstractmethod

class SimpleRealPSO(PSO):
    
    def __init__(self, target_function, intertia_weight_function, input_ranges: np.ndarray, swarm_size: int=100, c1: int=0.3, c2: int=0.4, iteration_count: int=100):
        self.N = swarm_size
        self.d = input_ranges.shape[0]

        super().__init__(target_function, input_ranges)

        self.w = intertia_weight_function
        self.c1 = c1
        self.c2 = c2
        self.iteration_count = iteration_count

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

