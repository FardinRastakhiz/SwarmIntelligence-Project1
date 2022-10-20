from ISwarmIntelligence import SwarmIntelligence
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
plt.rcParams["figure.autolayout"] = True


class SwarmDraw2D:

    def __init__(self, SIClass: SwarmIntelligence):
        self.si_class = SIClass
        self.fig, self.ax = plt.subplots()
        self.contours = self.create_function_contours()
        self.si_class.set_plot_utilities(self.fig, self.ax, self.contours)
        

        
    def create_function_contours(self):
        x = np.linspace(self.si_class.input_ranges[0, 0]-10, self.si_class.input_ranges[0, 1]+10, 200)
        y = np.linspace(self.si_class.input_ranges[1, 0]-10, self.si_class.input_ranges[1, 1]+10, 190)
        xv, yv = np.meshgrid(x, y)
        X = np.array([xv, yv])
        z = self.si_class.function(X)
        contours = self.ax.contour(x, y, z, 15)
        plt.colorbar(contours, ax=self.ax)
        return contours

    def draw(self):
        self.ax.set_xlim(self.si_class.input_ranges[0, 0]-5, self.si_class.input_ranges[0, 1]+5)
        self.ax.set_ylim(self.si_class.input_ranges[1, 0]-5, self.si_class.input_ranges[1, 1]+5)
        self.ax.plot(0, 0, 'o')

        animation = FuncAnimation(self.fig, func=self.si_class.run_iteration,
                                 frames=np.arange(0, self.si_class.iteration_count, 1), repeat=False, blit=False)
        plt.show()

