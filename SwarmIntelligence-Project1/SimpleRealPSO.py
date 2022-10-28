from ISwarmIntelligence import PSO
import numpy as np
from abc import ABC
from abc import abstractmethod
import matplotlib.pyplot as plt

class SimpleRealPSO(PSO):
    
    def __init__(self, target_function, intertia_weight_function, input_ranges: np.ndarray, swarm_size: int=100,
                c1: int=0.1, c2: int=0.1, iteration_count: int=100, is_maximization: bool = True):
        self.N = swarm_size
        self.d = input_ranges.shape[0]

        super().__init__(target_function, input_ranges, iteration_count, is_maximization)

        self.w = intertia_weight_function
        self.c1 = c1
        self.c2 = c2

        self.fitnesses : np.ndarray = self.evaluate_fitnesses()

        self.pbests = self.positions.copy()
        self.pbest_fitnesses = self.fitnesses.copy()
        self.gbest : np.ndarray = self.take_gbest()

    def run_iteration(self, i: int):
        self.__run_iteration_implementation(i)
        super().run_iteration(i)

    def run(self):
        for i in range(self.iteration_count):
            self.__run_iteration_implementation(i)
            self.best_fitnesses[i] = self.calculate_function(self.gbest.T)

        start_plot: int = 0
        plt.plot(np.arange(start_plot, self.iteration_count, 1), self.best_fitnesses[start_plot:])
        plt.xlabel("Iteration")
        plt.ylabel("Fitness (The more is better)")
        plt.title(f"Fitness diagram of SPO with {self.N} particle")
        plt.show()

    def __run_iteration_implementation(self, i: int):
        self.update_velocities(i)
        self.update_positions()
        self.fitnesses = self.evaluate_fitnesses()
        self.update_pbests()
        self.gbest = self.take_gbest()
        print(f'{i}: {self.fitnesses[0]}')

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
        return self.calculate_function(self.positions.T)

    def update_pbests(self):
        self.pbest_fitnesses = self.calculate_function(self.pbests.T)
        changes = self.fitnesses > self.pbest_fitnesses
        self.pbests[changes] = self.positions[changes]

    def take_gbest(self):
        m = self.pbests[np.argmax(self.pbest_fitnesses)]
        f = self.calculate_function(m.T)
        return m 

    def update_velocities(self, t: int):
        inertia = self.w(t) * self.velocities 
        cognitive = self.c1 * np.random.rand(self.N)[:, None] * (self.pbests - self.positions)
        social = self.c2 * np.random.rand(self.N)[:, None] * (self.gbest - self.positions)
        self.velocities = inertia + cognitive + social

    def update_positions(self):
        self.positions = self.positions + self.velocities
