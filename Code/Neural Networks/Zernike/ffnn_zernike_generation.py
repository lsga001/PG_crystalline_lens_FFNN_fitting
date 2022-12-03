import os
import numpy as np
import torch
import zernike
import torch.optim as optim
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.special import jv
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import time

def r_pol(n_max, Rho):
    mu_max = n_max
    #shape = Rho.shape()
    R = torch.zeros((n_max+1, 2*n_max+1, Rho.size(0)))
    R[0,0] = torch.ones_like(Rho)
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
    U = torch.zeros((n_max+1, m_max+1, Rho.size(0)))
    R = r_pol(n_max, Rho)
    for n_idx in np.arange(0, n_max+1):
        for m_idx in np.arange(0, m_max+1):
            mu_idx = n_idx - 2*m_idx
            #print(mu_idx)
            if (mu_idx > 0):
                U[n_idx, m_idx,] = R[n_idx, np.abs(mu_idx)]*np.sin(mu_idx*T.T)
            else:
                U[n_idx, m_idx,] = R[n_idx, np.abs(mu_idx)]*np.cos(mu_idx*T.T)
    return U

def cartesian_zpol(n_max, m_max, X, Y):
    U = np.zeros((n_max+1, m_max+1, X.size(0)))
    #U.reshape(1,-1)
    U[0,0,] = np.ones(np.shape(X.T))
    U[1,0,] = Y.T
    U[1,1,] = X.T

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
    return U.T

def zernike_pol(n,m, U, V, cartesian=True):
    if cartesian==True:
        S = cartesian_zpol(n, m, U, V)
    else:
        S = radial_zpol(n, m, U, V)
    return S

def test():
    res = 400
    R = torch.linspace(0, 1, res)
    T = torch.linspace(0, 2*np.pi, res)
    R, T = torch.meshgrid(R, T)
    r = torch.flatten(R, start_dim=0, end_dim=-1)
    t = torch.flatten(T, start_dim=0, end_dim=-1)

    n = 30
    m = n
    U = zernike_pol(n, m, r, t, cartesian=False).reshape(n+1,m+1, res, res)
    fig, ax = plt.subplots(subplot_kw = dict(projection='polar'))
    ax.pcolormesh(T, R, U[10,2])
    plt.plot()
    plt.show()

class NeuralNetwork(nn.Module):
    def __init__(self, sizes):
        super(NeuralNetwork, self).__init__()
        #self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.flatten = nn.Flatten()
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.stack = nn.Sequential()
        for i in np.arange(self.num_layers-2):
            self.stack.append(
                nn.Linear(self.sizes[i], self.sizes[i+1])
            )
            self.stack.append(
                #nn.Sigmoid()
                #nn.ReLU()
                nn.SELU()
            )
            self.stack.append(
                nn.AlphaDropout(0)
            )
        self.stack.append(
            nn.Linear(self.sizes[-2], self.sizes[-1])
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.stack(x)
        return logits



# check if gpu is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#test()

epochs = 1000
train_data_size = 50000
validation_data_size = 20000
LR = 0.05
MOM = 0.9
#inner_layer = [51, 47, 31, 43, 31, 47, 23]
inner_layer = [137, 109, 83, 47, 23, 11, 3, 2]



#x = torch.zeros((train_data_size+validation_data_size, 2), device=device)
#y = torch.zeros((train_data_size+validation_data_size, 1), device=device)
#x[:,0] = torch.randint(low=0, high=5, size=(train_data_size+validation_data_size,))
#x[:,1].uniform_(0,10)
#y = jv(x[:,0], x[:,1]).reshape(-1,1)

n_max = 4
m_max = n_max

x = torch.zeros((train_data_size+validation_data_size, 4), device=device)
y = torch.zeros((train_data_size+validation_data_size, 1), device=device)

x[:,0] = torch.randint(low=0, high=n_max+1, size=(train_data_size+validation_data_size,))
x[:,1] = torch.randint(low=0, high=m_max+1, size=(train_data_size+validation_data_size,))
x[:,2].uniform_(0, 1)
x[:,3].uniform_(0, 2*np.pi)

# "smart" approach
U = zernike_pol(n_max, m_max, x[:,2], x[:,3], cartesian=False)
for j in np.arange(train_data_size+validation_data_size):
    y[j] = U[int(x[j,0]), int(x[j,1]), j]

# naive approach
#for j in np.arange(train_data_size+validation_data_size):
#    print(j)
#    y[j] = zernike_pol(x[j,0], x[j,1], x[j,2], x[j,3], cartesian=False)[x[j,0], x[j,1]]


input_size = x.size(dim=1)
output_size = y.size(dim=1)
#output_size = 1

x_train = x[0:train_data_size]
y_train = y[0:train_data_size]
x_valid = x[train_data_size:-1]
y_valid = y[train_data_size:-1]

layers = inner_layer
layers.append(output_size)
layers.insert(0, input_size)

net = NeuralNetwork(layers).to(device)
params = list(net.parameters())

# create optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOM) # ReLU
#optimizer = optim.SGD(net.parameters(), lr=0.5) # Sigmoid

n = train_data_size
bs = int(n/10) # batch size

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs*2)

for epoch in np.arange(epochs):
    net.train()
    for xb, yb in train_dl:
        pred = net(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    net.eval()
    with torch.no_grad():
        valid_loss = sum(criterion(net(xv), yv) for xv, yv in valid_dl)
    print("epoch: ", epoch, valid_loss / len(valid_dl))

#X0 = torch.zeros((100, input_size), device=device)
#Y0 = torch.zeros((100, output_size), device=device)

# save parameters
SAVE_PATH = "./saved/model1.pt"
torch.save(net.state_dict(), SAVE_PATH)

# load parameters
eval_net = NeuralNetwork(layers)
eval_net.load_state_dict(torch.load(SAVE_PATH))
eval_net.eval()


## Visualization of results

# generate domain in unit disc
res = 500
domain_r = torch.linspace(0, 1, res)
domain_t = torch.linspace(0, 2*np.pi, res)

# make grid
domain_R, domain_T = torch.meshgrid(domain_r, domain_t)

## specify zernike pol to visualize if at all
#n = 2
#m = 0

flat_R = torch.flatten(domain_R)
flat_T = torch.flatten(domain_T)

inp = torch.zeros(res**2, 4)
inp[:,2] = flat_R
inp[:,3] = flat_T



vmin = -1
vmax = 1
cmap_name = 'viridis'
cmap = cm.get_cmap(cmap_name)
normalizer = Normalize(vmin,vmax)
im = cm.ScalarMappable(cmap=cmap, norm=normalizer)

fig, ax = plt.subplots(n_max+3, m_max+3, subplot_kw = dict(projection='polar'))

for n in np.arange(n_max+3):
    for m in np.arange(m_max+3):
        ax[n,m].set_aspect('equal', adjustable='box')
        ax[n,m].grid(visible=None)
        if (m<=n):
            inp[:,0] = torch.zeros_like(flat_R) + n
            inp[:,1] = torch.zeros_like(flat_R) + m
            with torch.no_grad():
                start = time.time()
                P = eval_net(inp).reshape(res, res)
                end = time.time()
                print(end-start)

            ax[n,m].pcolormesh(domain_T, domain_R, P, cmap=cmap, norm=normalizer)
        else:
            ax[n,m].axis('off')
        ax[n,m].set_xticklabels([])
        ax[n,m].set_yticklabels([])
fig.colorbar(im, ax = ax.ravel().tolist())
plt.show()
