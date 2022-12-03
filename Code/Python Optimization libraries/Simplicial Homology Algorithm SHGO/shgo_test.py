import numpy as np
from scipy.optimize import rosen, shgo
import time

def f_cost(X):
    return np.sin(X[0])*np.exp((1-np.cos(X[1]))**2)\
        + np.cos(X[1])*np.exp((1-np.sin(X[0]))**2)\
        + (X[0] - X[1])**2

bounds = [(-2*np.pi, 2*np.pi), (-2*np.pi, 2*np.pi)]

start1 = time.time()
result1 = shgo(f_cost, bounds, n=2**10, iters = 3, sampling_method = 'simplicial')
end1 = time.time()

start2 = time.time()
result2 = shgo(f_cost, bounds, iters = 3, sampling_method = 'sobol')
end2 = time.time()

print(result1.x)
print(result1.fun)
print("Time of execution of simplicial (iters=3) is: ", end1-start1)
print(result2.x)
print(result2.fun)
print("Time of execution of sobol (iters=3) is: ", end2-start2)
