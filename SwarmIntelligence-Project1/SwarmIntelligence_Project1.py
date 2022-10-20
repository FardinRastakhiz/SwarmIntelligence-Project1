from SimpleRealPSO import SimpleRealPSO
from DrawPlot import SwarmDraw2D
import numpy as np


def main():
    iteration_count=1000
    weight_function = lambda t:  0.9 - (0.7 / iteration_count) * t

    #target_function = lambda X: 3 * X[0] ** 2 - 2 * X[0] * X[1] + 3 * X[1] ** 2 - X[0] - X[1]
    #input_ranges = np.array([[-5, 5], [-5, 5]])
    
    target_function = lambda X: (X[0]- 3.14) ** 2 + (X[1]-2.72) ** 2 +np.sin(3*X[0]+1.41)-np.sin(4*X[1]-1.73)
    input_ranges = np.array([[0, 5], [0, 5]])

    spso = SimpleRealPSO(target_function, weight_function, input_ranges,swarm_size=15, iteration_count=iteration_count, is_maximization=False)
    

    draw_plot = False
    if(draw_plot): # Slow
        plot_drawer = SwarmDraw2D(spso)
        plot_drawer.draw()
    else: # Fast
        spso.run()

if __name__ == "__main__":
    main()
