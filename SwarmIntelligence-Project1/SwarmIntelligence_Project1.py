from SimpleRealPSO import SimpleRealPSO
from DrawPlot import SwarmDraw2D
import numpy as np


def main():
    iteration_count = 1000
    target_function = lambda X: 3 * X[0] ** 2 - 2 * X[0] * X[1] + 3 * X[1] ** 2 - X[0] - X[1]
    weight_function = lambda t:  0.9 - (0.7 / iteration_count) * t
    input_ranges = np.array([[-5, 5], [-5, 5]])
    spso = SimpleRealPSO(target_function, weight_function, input_ranges, iteration_count)
    spso.run()

    draw_plot = SwarmDraw2D(spso)
    draw_plot.draw()

if __name__ == "__main__":
    main()
