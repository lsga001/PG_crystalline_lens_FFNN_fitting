import numpy as np
import matplotlib as plt

# Initialization

# Selection

# Genetic operators

# 















#
#
## Parameters
#
#epochs = 100
#N = 100 # Size of population
#Gamma = 0.5 # Mutation factor
#dGamma = 0.99 # Rate of mutation factor decrease
#
## Restrictions (Bounds)
#lb = -100
#ub = 100
#nvars = 2
#
#
#
#
#### functions ###
#
#def benchmark_fun1():
#    return None
#
#def fun_Order(x, y, lb, ub):
#    D = np.size(x, 1) # length of each vector/solution
#
#    # Probabilidades
#    Prob_y = y/(np.sum(y) + 1 * (sum(y)==0))\
#    + (1/np.size(x,0)) * (np.sum(y)==0)
#
#    # LLenamos la matriz IndividuosBinarios
#    Matriz = np.zeros((np.size(x,0), np.size(x,1)+1))
#    Matriz[:, 0:D-1] = x
#    Matriz[:,D] = (Prob_y)
#
#    Condition1 = np.any(x<lb, 2)
#    Condition2 = np.any(x>ub, 2)
#    Matriz[Condition1, D] = 0
#    Matriz[Condition2, D] = 0
#
#    # Acomoda la matriz y realiza una suma acumulada para sus probabilidades respectivas
#    O = np.lexsort(Matriz, D)
#    O[:,D] = np.cumsum(O[:,D])
#    return O
#
#def fun_Selection(O):
#    # Obtenemos un par de indices aleatorios basados en las probabilidades
#    p = np.random.choice(np.size(O,0), 2, True, O[:,-1])
#
#    # Padres fuertes
#    Father = O[p[0], 0:-2]
#    Mother = O[p[1], 0:-1]
#    return (Father, Mother)
#
#def fun_Mix(Father, Mother):
#    # Blender method
#
#    # Find min and max in each component
#    # and make vectors with min and max for bounds
#    x1 = np.min([Father, Mother], [], 1)

