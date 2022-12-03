import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.random import default_rng

# randomness
rng = default_rng()

# cost function

def f_distanceAtoB(A, B):
    distance = np.sqrt(np.vdot(B-A, B-A))
    return distance

def f_costDistance(ant_Path):
    total_Distance = 0
    local_Distance = 0
    local_Displacement = ant_Path[0,]*0
    for i in np.arange(1,np.size(ant_Path,0)):
        local_Displacement = ant_Path[i,] - ant_Path[i-1,]
        local_Distance = np.sqrt(np.vdot(local_Displacement,local_Displacement))
        total_Distance = total_Distance + local_Distance
    local_Displacement = ant_Path[0,] - ant_Path[-1,]
    local_Distance = np.sqrt(np.vdot(local_Displacement,local_Displacement))
    total_Distance = total_Distance + local_Distance
    return total_Distance

# Get all the coordinates of each city

filename = 'xqf131.tsp'
skiprows_n = 8
which_cols = (1,2)

coordinates_Cities = np.loadtxt(filename, \
                                delimiter=' ', \
                                skiprows=skiprows_n,\
                                usecols=which_cols)

num_Cities = np.size(coordinates_Cities)


def f_Probabilities(i, G, Cand_Gidx, T, alpha, beta):
    d = f_distanceAtoB
    num_Cand_Gidx = np.size(Cand_Gidx,0)
    Pi = np.zeros((num_Cand_Gidx,))
    numer = 0
    denom = 0

    for r in np.arange(np.size(Cand_Gidx,0)):
        l = Cand_Gidx[r]
        denom = denom + (T[i,l])**(alpha) * (1/d(G[i,], G[l,]))**(beta)

    for m in np.arange(np.size(Cand_Gidx,0)):
        j = Cand_Gidx[m]
        numer = (T[i,j])**(alpha) * (1/d(G[i,], G[j,]))**(beta)
        Pi[m] = numer / denom
    return Pi

def f_ACO(G, alpha, beta, rho, tao_0, Q, N):
    epochs = 10
    num_Vertex = np.size(G,0)
    antPos_idx = np.zeros((N, num_Vertex), dtype=np.int64)
    Tao = np.zeros((num_Vertex,num_Vertex)) + tao_0
    l_k = np.zeros((epochs,N))
    bestSolution = np.random.permutation(num_Vertex) # dummy solution
    bestCost = f_costDistance(bestSolution) # dummy cost
    for t in np.arange(epochs):
        for k in np.arange(N):
            Cand_Gidx = np.arange(num_Vertex)
            # Choose random initial index
            # and put ant j in random vertex
            antPos_idx[k,0] = rng.integers(low=0, high=num_Vertex, size=1)

            # remove ant_Pos from Cand
            Cand_Gidx = np.delete(Cand_Gidx, np.where(Cand_Gidx == antPos_idx[k,0]))

            for i in np.arange(1, num_Vertex):
                # get probabilities vector from i to other Cand
                Pi = f_Probabilities(antPos_idx[k,i-1], G, Cand_Gidx, Tao, alpha, beta)

                # get next vertex index
                j = np.random.choice(Cand_Gidx, 1, p=Pi)

                # set next vertex
                antPos_idx[k,i] = j

                # remove vertex from candidate list
                Cand_Gidx = np.delete(Cand_Gidx, np.where(Cand_Gidx == j))

        # pheromone evaporation
        Tao = Tao * (1-rho)

        # get cycle evaluation and update edges
        for k in np.arange(N):
            l_k[t, k] = f_costDistance(antPos_idx[k,])
            for m in np.arange(1, num_Vertex):
                i = antPos_idx[k,m-1]
                j = antPos_idx[k,m]
                Tao[i,j] = Tao[i,j] + Q/(l_k[t,k])
                Tao[j,i] = Tao[i,j]
            Tao[antPos_idx[k,-1],antPos_idx[k,0]] = Tao[antPos_idx[k,-1],antPos_idx[k,0]] + Q/(l_k[t,k])
            Tao[antPos_idx[-1,k],antPos_idx[0,k]] = Tao[antPos_idx[k,k-1],antPos_idx[k,0]]
            if (l_k[t,k] < bestCost):
                bestSolution = antPos_idx[k,:]
                bestCost = f_costDistance(bestSolution)
                #print(bestCost)

    return bestSolution, bestCost


tao_0 = 1
alpha = 1
beta = 1
rho = 0.1
N = 10
Q = 1

bestSolution, bestCost = f_ACO(coordinates_Cities, alpha, beta, rho, tao_0, Q, N)

print(bestSolution)
print(bestCost)
