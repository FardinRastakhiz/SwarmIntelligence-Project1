from ISwarmIntelligence import SwarmIntelligence
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
plt.rcParams["figure.autolayout"] = True

class SwarmDraw2D:

    def __init__(self, SIClass: SwarmIntelligence):
        self.si_class = SIClass
        self.ranges = self.si_class.input_ranges
        self.function = self.si_class.function
        self.positions = self.si_class.positions
        self.velocities = self.si_class.velocities
        self.fig, self.ax = plt.subplots()
        self.x_data = []
        self.y_data = []
        self.contours = None

    def draw(self):

        self.ax.set_xlim(self.si_class.input_ranges[0, 0], self.si_class.input_ranges[0, 1])
        self.ax.set_ylim(self.si_class.input_ranges[1, 0], self.si_class.input_ranges[1, 1])
        x = np.linspace(self.si_class.input_ranges[0, 0], self.si_class.input_ranges[0, 1], 100)
        y = np.linspace(self.si_class.input_ranges[1, 0], self.si_class.input_ranges[1, 1], 99)
        xv, yv = np.meshgrid(x, y)
        X = np.array([xv, yv])
        z = self.si_class.function(X)
        self.ax.plot(0, 0, 'o')
        self.contours = self.ax.contour(x, y, z, 50)
        plt.colorbar(self.contours, ax=self.ax)

        animation = FuncAnimation(self.fig, func=self.animation_function,
                                 frames=np.arange(0, 10, 0.1), interval=10, blit=False)
        plt.show()

        
    def animation_function(self, i):
        self.ax.clear()
        self.x_data.append(i*10)
        xx = np.array(self.x_data)
        yy = np.random.rand(xx.shape[0],) * i
        self.y_data.append(i)
        for collection in self.contours.collections:
            self.ax.add_collection(collection)
        self.ax.plot(xx, yy, 'o')





from ISwarmIntelligence import SwarmIntelligence
from SimpleRealPSO import SimpleRealPSO
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
plt.rcParams["figure.autolayout"] = True

iteration_count = 1000
target_function = lambda X: 3 * X[0] ** 3 - 2 * X[0] * X[1] + 3 * X[1] ** 2 - X[0] - X[1]
weight_function = lambda t:  0.9 - (0.7 / iteration_count) * t
input_ranges = np.array([[-5, 5], [-5, 5]])
si_class = SimpleRealPSO(target_function, weight_function, input_ranges, iteration_count)
ranges = si_class.input_ranges
function = si_class.function
positions = si_class.positions
velocities = si_class.velocities
fig, ax = plt.subplots()
x_data = []
y_data = []
contours = None

def animation_function(i):
    ax.clear()
    x_data.append(i*10)
    xx = np.array(x_data)
    yy = np.random.rand(xx.shape[0],) * i
    y_data.append(i)
    for collection in contours.collections:
        ax.add_collection(collection)
    ax.plot(xx, yy, 'o')
    ax.quiver(xx,yy, xx,yy, units='width')


ax.set_xlim(si_class.input_ranges[0, 0], si_class.input_ranges[0, 1])
ax.set_ylim(si_class.input_ranges[1, 0], si_class.input_ranges[1, 1])
x = np.linspace(si_class.input_ranges[0, 0], si_class.input_ranges[0, 1], 100)
y = np.linspace(si_class.input_ranges[1, 0], si_class.input_ranges[1, 1], 99)
xv, yv = np.meshgrid(x, y)
X = np.array([xv, yv])
z = si_class.function(X)
ax.plot(0, 0, 'o')
contours = ax.contour(x, y, z, 50)
plt.colorbar(contours, ax=ax)

animation = FuncAnimation(fig, func=animation_function,
                            frames=np.arange(0, 10, 0.1), interval=10, blit=False)

plt.show()
