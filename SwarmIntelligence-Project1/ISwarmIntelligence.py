from abc import ABC 
from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt

class SwarmIntelligence(ABC):
    def __init__(self, target_function, input_ranges, iteration_count: int=100, is_maximization: bool = True):
        self.function = target_function
        self.input_ranges = input_ranges
        self.positions : np.ndarray = self.generate_swarm_positions()
        self.velocities : np.ndarray = self.generate_swarm_velocities()
        self.iteration_count = iteration_count
        self.best_fitnesses = np.zeros((self.iteration_count,))
        self.is_maximization = 1 if is_maximization else -1

        self.fig = None
        self.ax = None
        self.contours = None

    def set_plot_utilities(self, fig, ax, contours):
        self.fig, self.ax, self.contours = fig, ax, contours

    @abstractmethod
    def run_iteration(self, i: int):
        self.ax.clear()
        xx = self.positions[:, 0]
        yy = self.positions[:, 1]
        vv = self.velocities[:, 0]
        uu = self.velocities[:, 1]
        for collection in self.contours.collections:
            self.ax.add_collection(collection)
        self.ax.plot(xx, yy, 'o')
        self.ax.quiver(xx, yy, vv, uu, units='width')

    def calculate_function(self, X):
        return self.is_maximization * self.function(X)

        
class PSO(SwarmIntelligence):
    pass