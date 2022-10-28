from ISwarmIntelligence import PSO
import numpy as np
from abc import ABC
from abc import abstractmethod
import matplotlib.pyplot as plt

class SimpleBinaryPSO(PSO):
    
    def __init__(self, target_function, intertia_weight_function, input_ranges: np.ndarray,
                input_lengths: int=8, swarm_size: int=100,
                c1: int=0.1, c2: int=0.1, iteration_count: int=100, is_maximization: bool=True):
        self.N = swarm_size
        self.input_lengths = input_lengths
        self.varialbes_count = input_ranges.shape[0]
        self.d = input_ranges.shape[0]
        self.particle_length = input_ranges.shape[0] * input_lengths
        self.binary_to_decimal = (2 ** np.array([i for i in range(input_lengths - 1, -1, -1)])).T
        self.max_power_length = 2 ** input_lengths - 1

        super().__init__(target_function, input_ranges, iteration_count, is_maximization)
        

        self.w = intertia_weight_function
        self.c1 = c1
        self.c2 = c2

        self.fitnesses : np.ndarray = self.evaluate_fitnesses()

        self.pbests = self.positions_binary.copy()
        self.pbest_fitnesses = self.fitnesses.copy()
        self.gbest : np.ndarray = self.take_gbest()

    def run_iteration(self, i: int):
        self.__run_iteration_implementation(i)
        super().run_iteration(i)

    def run(self):
        for i in range(self.iteration_count):
            self.__run_iteration_implementation(i)
            gbest_real = np.squeeze(self.get_real_values(self.gbest))
            fitness_value = self.calculate_function(gbest_real.T)
            self.best_fitnesses[i] = fitness_value

        start_plot : int = 0
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
        self.positions_binary = np.random.rand(self.N, self.particle_length)
        self.positions_binary[self.positions_binary >= 0.5] = 1
        self.positions_binary[self.positions_binary < 0.5] = 0
        return self.get_real_values(self.positions_binary)

    def get_real_values(self, binary_array):
        if len(binary_array.shape)==1: 
            binary_array = binary_array[None, :]
        real_array = np.zeros((binary_array.shape[0], self.varialbes_count))
        for i in range(self.varialbes_count):
            parameter = binary_array[:, range(i * self.input_lengths, (i + 1) * self.input_lengths)]
            normalized = np.matmul(parameter, self.binary_to_decimal)
            normalized = normalized / self.max_power_length
            real_array[:, i] = self.input_ranges[i,0] + (self.input_ranges[i,1] - self.input_ranges[i,0]) * normalized
        return real_array
            
    def generate_swarm_velocities(self):
        velocities = np.random.rand(self.N, self.particle_length) * 2
        #lower_bound = self.input_ranges[:, 0][None, :]
        #upper_bound = self.input_ranges[:, 1][None, :]
        #velocities = ((upper_bound - lower_bound) / 2.0) * velocities
        return velocities

    def evaluate_fitnesses(self):
        return self.calculate_function(self.positions.T)

    def update_pbests(self):
        pbests_real = self.get_real_values(self.pbests)
        self.pbest_fitnesses = self.calculate_function(pbests_real.T)
        changes = self.fitnesses > self.pbest_fitnesses
        self.pbests[changes] = self.positions_binary[changes]

    def take_gbest(self):
        m = self.pbests[np.argmax(self.pbest_fitnesses)]
        #real_m = get_real_values(m)
        #f = self.calculate_function(real_m.T)
        return m 

    def update_velocities(self, t: int):
        inertia = self.w(t) * self.velocities 
        cognitive = self.c1 * np.random.rand(self.N)[:, None] * (self.pbests - self.positions_binary)
        social = self.c2 * np.random.rand(self.N)[:, None] * (self.gbest - self.positions_binary)
        self.velocities = inertia + cognitive + social

    def update_positions(self):
        p = 1 / (1 + np.exp(-self.velocities))
        rands = np.random.rand(p.shape[0], p.shape[1])
        changes = rands > p
        self.positions_binary[changes] = 1 - self.positions_binary[changes]
        self.positions = self.get_real_values(self.positions_binary)
