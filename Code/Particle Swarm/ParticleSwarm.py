import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.random import default_rng

# randomness
rng = default_rng()

# cost functions
def f_cost1(x):
    X = np.meshgrid(*x)
    return -200*np.exp(-0.02*np.sqrt(X[0]**2 + X[1]**2))

def f_cost2(x):
    X = np.meshgrid(*x)
    return np.cos(X[0])*np.sin(X[1]) - X[0]/(X[1]**2 + 1)

def f_cost3(x):
    X = np.meshgrid(*x)
    P = 1
    for i in np.arange(np.size(x,0)):
        P = P * np.sqrt(X[i])*np.sin(X[i])
    return -P

def f_cost4(x):
    X = np.meshgrid(*x)
    return np.sin(X[0])*np.exp((1-np.cos(X[1]))**2)\
        +np.cos(X[1])*np.exp((1-np.sin(X[0]))**2)\
        +(np.subtract(X[0],X[1]))**2

def f_cost5(x):
    X = np.meshgrid(*x)
    return X[0]**2 - 12*X[0] + 11 \
        + 10*np.cos(np.pi*X[0]/2) + 8*np.sin(5*np.pi*X[0]/2)\
        - (1/5)**(0.5)*np.exp(-0.5*(X[1]-0.5)**2)

def f_cost6(x):
    X = np.meshgrid(*x)
    return 0.5 + (np.cos(np.sin(X[0]**2 + X[1]**2))**2 - 0.5)\
        / (1 + 0.001*(X[0]**2 + X[1]**2)**2)

# Setup

w = 1/(2*np.log(2))
cp = 0.5+np.log(2)
cg = 0.5+np.log(2)


dim = 2 # dimension of solution vector / number of parameters
t_N = 100 # number of iterations


#f_c, xmin, xmax = f_cost1, [-10, -10], [10, 10]

#f_c, xmin, xmax = f_cost2, [-1, -1], [2, 1]

#f_c, xmin, xmax = f_cost3, [0, 0], [10, 10]

f_c, xmin, xmax = f_cost4, [-2*np.pi, -2*np.pi], [2*np.pi, 2*np.pi]

#f_c, xmin, xmax = f_cost5, [-30, -30], [30, 30]

#f_c, xmin, xmax = f_cost6, [-100, -100], [100, 100]


N = 10 # particles


def ParticleSwarmOptimization(f, dim, N, t_N, w, cp, cg, xmin, xmax):
    # Initialization of initial positions

    X = np.zeros((t_N, N, dim)) # Initialize t_N solution steps with size dim for N particles
    V = np.zeros((t_N, N, dim)) # Initialize t_N velocity steps with size dim for N particles
    P = np.zeros((t_N, N, dim)) # Initialize t_N personal best steps with size dim for N particles
    G = np.zeros((t_N, 1, dim)) # Initialize t_N global best steps with size dim

    # Set random solutions at t=0
    for i in np.arange(dim):
        X[0,:,i] = rng.uniform(xmin[i], xmax[i], (1,N))

    # Calculate initial personal best and global best

    P[0,:,:] = X[0,:,:]
    G[0,0,:] = X[0,0,:]

    for i in np.arange(N):
        if (f(X[0,i,:]) < f(G[0,0,:])):
            G[0,0,:] = X[0,i,:]

    # Iteration

    Gm = G[0,0,:]
    for t in np.arange(1,t_N):
        for i in np.arange(N):
            rp = rng.uniform(0,1,1)
            rg = rng.uniform(0,1,1)
            V[t,i,:] = w*V[t-1,i,:] + cp*rp*(P[t-1,i,:] - X[t-1,i,:]) + cg*rg*(G[t-1,0,:] - X[t-1,i,:])
            X[t,i,:] = X[t-1,i,:] + V[t,i,:]
            if ((X[t,i,:] <= xmax).all() & (X[t,i,:] >= xmin).all()):
                if (f(X[t,i,:]) < f(P[t-1,i,:])):
                    P[t,i,:] = X[t,i,:]
                    if (f(P[t,i,:]) < f(Gm)):
                        Gm = P[t,i,:]
                else:
                    P[t,i,:] = P[t-1,i,:]
            else:
                X[t,i,:] = X[t-1,i,:]
                V[t,i,:] = -V[t-1,i,:]
                #V[t,i,:] = 0
                P[t,i,:] = P[t-1,i,:]
        #w = w - 1/t_N*0.6
        #print(w)
        G[t,0,:] = Gm
    return X, V, P, G

# Evolution
start = time.time()
S, V, P, G = ParticleSwarmOptimization(f_c, dim, N, t_N, w, cp, cg, xmin, xmax)
end = time.time()

# Visualization

xmin = np.min(xmin)
xmax = np.max(xmax)
x = np.linspace(xmin, xmax, 1000)
y = np.linspace(xmin, xmax, 1000)

Z = f_c([x, y])

print(G[-1,0,:])
f_g = f_c([G[-1,0,0], G[-1,0,1]])
print(f_g[0,0])
print(end-start)

X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots()
im = ax.pcolormesh(X, Y, Z)
ax.plot(S[0,:,0], S[0,:,1], '.b')
ax.plot(S[-1,:,0], S[-1,:,1], '.r')
ax.plot(G[:,0,0], G[:,0,1], '.w')
ax.set_xlim(xmin, xmax)
ax.set_ylim(xmin, xmax)
ax.set_box_aspect(1)

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0)

plt.colorbar(im, cax=cax)


fig2, ax2 = plt.subplots()

im2 = ax2.pcolormesh(X, Y, Z)
test = ax2.scatter(np.reshape(S[0,:,0],(N,)), np.reshape(S[0,:,1],(N,)), 50, 'w', '.')
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(xmin, xmax)
ax2.set_box_aspect(1)



def update(frame_number):
    t = frame_number % t_N
    test.set_offsets(np.reshape(S[t,:,:],(N,2)))

# make animation
anim = FuncAnimation(fig2, update, interval=1)

# saving to m4 using ffmpeg writer
writervideo = animation.FFMpegWriter(fps=10)
anim.save('PSO.mp4', writer=writervideo)
plt.close()

plt.show()
