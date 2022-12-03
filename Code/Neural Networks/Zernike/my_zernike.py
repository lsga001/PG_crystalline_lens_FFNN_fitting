import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import cm
from matplotlib.colors import Normalize
from zernike import RZern

def r_pol(n_max, Rho):
    mu_max = n_max
    shape = np.shape(Rho)
    R = np.zeros((n_max+1, 2*n_max+1, np.size(Rho,0), np.size(Rho,1)))
    R[0,0] = np.ones(shape)
    R[1,1] = Rho

    for n_idx in np.arange(2, n_max+1):
        for mu_idx in np.arange(0, n_idx+1):
            if (mu_idx==n_idx):
                R[n_idx, mu_idx] = Rho*R[n_idx-1,n_idx-1]
            elif (mu_idx==0):
                R[n_idx, mu_idx] = 2*Rho*R[n_idx-1,1]-R[n_idx-2,0]
            else:
                R[n_idx, mu_idx] = Rho*(R[n_idx-1,mu_idx-1] + R[n_idx-1, mu_idx+1]) - R[n_idx-2,mu_idx]
    return R

def radial_zpol(n_max, m_max, Rho, T):
    U = np.zeros((n_max+1, m_max+1, np.size(Rho,0), np.size(Rho,1)))
    R = r_pol(n_max, Rho)
    for n_idx in np.arange(0, n_max+1):
        for m_idx in np.arange(0, m_max+1):
            mu_idx = n_idx - 2*m_idx
            print(mu_idx)
            if (mu_idx > 0):
                U[n_idx, m_idx,] = R[n_idx, np.abs(mu_idx)]*np.sin(mu_idx*T)
            else:
                U[n_idx, m_idx,] = R[n_idx, np.abs(mu_idx)]*np.cos(mu_idx*T)
    return U

def cartesian_zpol(n_max, m_max, X, Y):
    U = np.zeros((n_max+1, m_max+1, np.size(X,0), np.size(X,1)))
    U[0,0,] = np.ones(np.shape(X))
    U[1,0,] = Y
    U[1,1,] = X

    for n_idx in np.arange(2, n_max+1):
        for m_idx in np.arange(0, n_idx+1):
            if (m_idx==0):
                U[n_idx, m_idx,] = X*U[n_idx-1, 0,] \
                    + Y*U[n_idx-1,n_idx-1,]
            elif (m_idx==n_idx):
                U[n_idx, m_idx,] = X*U[n_idx-1,n_idx-1,] \
                    - Y*U[n_idx-1,0,]
            elif (n_idx%2==1) & (m_idx==((n_idx-1)/2)):
                U[n_idx, m_idx,] = Y*U[n_idx-1,n_idx-1-m_idx,]\
                    + X*U[n_idx-1, m_idx-1,]\
                    - Y*U[n_idx-1, n_idx-m_idx,]\
                    - U[n_idx-2,m_idx-1,]
            elif (n_idx%2==1) & (m_idx==((n_idx-1)/2 + 1)):
                U[n_idx, m_idx,] = X*U[n_idx-1,m_idx,]\
                    + Y*U[n_idx-1, n_idx-1-m_idx,]\
                    + X*U[n_idx-1, m_idx-1,]\
                    - U[n_idx-2,m_idx-1,]
            elif (n_idx%2==0) & (m_idx == (n_idx/2)):
                U[n_idx, m_idx,] = 2*X*U[n_idx-1,m_idx,]\
                    + 2*Y*U[n_idx-1, m_idx-1,]\
                    - U[n_idx-2, m_idx-1,]
            else:
                U[n_idx, m_idx,] = X*U[n_idx-1,m_idx,]\
                    + Y*U[n_idx-1,n_idx-1-m_idx,]\
                    + X*U[n_idx-1,m_idx-1,]\
                    - Y*U[n_idx-1,n_idx-m_idx,]\
                    - U[n_idx-2, m_idx-1,]
    return U

def zernike_pol(n,m, U, V, cartesian=True):
    if cartesian==True:
        S = cartesian_zpol(n, m, U, V)
    else:
        S = radial_zpol(n, m, U, V)
    return S

n_max = 6
m_max = n_max

res = 2**9
x = np.linspace(-1, 1, res)
y = np.linspace(-1, 1, res)

X, Y = np.meshgrid(x, y)

r = np.linspace(0, 1, res)
theta = np.linspace(0, 2*np.pi, res)

R, T = np.meshgrid(r, theta)

X = R*np.cos(T)
Y = R*np.sin(T)

start1 = time.time()
U = zernike_pol(n_max, m_max, X, Y, cartesian=True)
#U = zernike_pol(n_max, m_max, R, T, cartesian=False)
start2 = time.time()
# Zernike package
cart = RZern(5) # takes n_max as argument?
cart.make_cart_grid(X, Y)

c = np.zeros(cart.nk)
print(np.shape(c))
c[7] = 1

Phi = cart.eval_grid(c, matrix=True)
end = time.time()


print(start2 - start1)
print(end - start2)

vmin = np.min(U)
vmax = np.max(U)

#vmin = -1
#vmax = 1


fig, ax = plt.subplots()

ax.pcolormesh(X, Y, Phi/(np.sqrt((2)*(3+1)))-U[3,2,:,:])#, vmin=-1, vmax=1)
ax.set_aspect('equal', adjustable='box')
plt.show()

# Visualization
cmap_name = 'viridis'
cmap = cm.get_cmap(cmap_name)
normalizer = Normalize(vmin,vmax)
im = cm.ScalarMappable(cmap=cmap, norm=normalizer)

fig, ax = plt.subplots(n_max, n_max, subplot_kw=dict(projection='polar'))

for n in np.arange(n_max):
    for m in np.arange(n_max):
        ax[n,m].set_aspect('equal', adjustable='box')
        ax[n,m].grid(visible=None)
        if (m<= n):
            ax[n,m].pcolormesh(T, R, U[n,m,:,:], cmap=cmap, norm=normalizer)
        else:
            ax[n,m].axis('off')
        ax[n,m].set_xticklabels([])
        ax[n,m].set_yticklabels([])
        #fig.add_axes([])
fig.colorbar(im, ax = ax.ravel().tolist())
#plt.savefig("First21ZernikePolynomials_%s.png"%cmap_name, dpi=600)
plt.show()
