import numpy as np
import torch


def n_index(x, y, z, m, b, az, ar):
    Nidx = torch.zeros_like(x)

# "Constants"
b = 0
az = 0
V = 0 # [mm]^3
ar = 0

# Window
limits = np.array([[-1, 1], [-1, 1], [0, 10]])

# Resolution
Nx = 2**13+1
Ny = 2**13+1
Nz = 2**10+1

# Mesh
x = np.linspace(limits[0,0], limits[0,1], Nx)
y = np.linspace(limits[1,0], limits[1,1], Ny)
z = np.linspace(limits[2,0], limits[2,1], Nz)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# User variables
m = 2.8
