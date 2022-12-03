import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
from pyswarms.utils.plotters.formatters import Mesher, Designer
import time



def f_cost(X):
    """ Takes input with shape (n_particles, dimensions)"""
    return np.sin(X[:,0])*np.exp((1-np.cos(X[:,1]))**2)\
        + np.cos(X[:,1])*np.exp((1-np.sin(X[:,0]))**2)\
        + (X[:,0] - X[:,1])**2

# set up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
# define boundaries
constraints = (np.array([-2*np.pi, -2*np.pi]), np.array([2*np.pi,2*np.pi]))
# call instance of PSO
start = time.time()
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=constraints)
# perform optimization
best_cost, best_pos = optimizer.optimize(f_cost, iters=100)
end = time.time()

### Visualization ###

print(best_pos)
print(best_cost)
print(f_cost(np.reshape(best_pos,(1,2)))[0])
print(end-start)

# Plot the sphere function's mesh for better plots
#m = Mesher(func=f_cost,
#           limits=[(-1, 1), (-1, 1)])
#d = Designer(limits=[(-1, 1), (-1, 1)])
#m = Mesher(func=f_cost)
# Adjust figure limits
#animation = plot_contour(pos_history=optimizer.pos_history, mesher=m, designer=d, mark=(0,0))
# Enables us to view it in a Jupyter notebook
##animation.save('plot0.gif', writer='imagemagick', fps=10)
#plt.show()
