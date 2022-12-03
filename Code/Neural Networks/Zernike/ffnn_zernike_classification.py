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
    R = torch.zeros((n_max+1, 2*n_max+1, Rho.size(0), Rho.size(1)))
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
    U = torch.zeros((n_max+1, m_max+1, Rho.size(0), Rho.size(1)))
    R = r_pol(n_max, Rho)
    for n_idx in np.arange(0, n_max+1):
        for m_idx in np.arange(0, m_max+1):
            mu_idx = n_idx - 2*m_idx
            #print(mu_idx)
            if (mu_idx > 0):
                U[n_idx, m_idx,] = R[n_idx, np.abs(mu_idx)]*np.sin(mu_idx*T)
            else:
                U[n_idx, m_idx,] = R[n_idx, np.abs(mu_idx)]*np.cos(mu_idx*T)
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

#def generate_ZernikeComb(n_max, n_train, width, height):
#    

def rc(n):
    # rc(1) = (1, 1); rc(33) -> (8, 5)
    assert n > 0 and int(n) == n
    sum = 0
    row = 0
    while sum < n:
        row += 1
        sum += row
    col = n - sum + row
    return row, col

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
        self.stack_conv = nn.Sequential()
        self.stack_trans = nn.Sequential()
        self.stack_lin = nn.Sequential()

        # assume input of size 300 x 300

        self.conv_channels = [1, 8, 16, 32]

        for i in np.arange(len(self.conv_channels)-1):
            self.stack_conv.append(
                nn.Conv2d(self.conv_channels[i],
                          self.conv_channels[i+1],
                          kernel_size=3, stride=1, padding=0)
            )
            self.stack_conv.append(
                #nn.ReLU()
                nn.SELU()
            )
            self.stack_conv.append(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            )

        last_size = 12
        self.stack_trans.append(
            nn.AdaptiveAvgPool2d((last_size,last_size))
        )
        self.stack_trans.append(
            nn.Flatten(start_dim=1, end_dim=-1)
        )

        self.stack_lin.append(
            nn.Linear(self.conv_channels[-1]*last_size**2, self.sizes[0])
        )

       # self.stack.append(nn.Flatten())

        for i in np.arange(self.num_layers-2):
            self.stack_lin.append(
                nn.Linear(self.sizes[i], self.sizes[i+1])
            )
            self.stack_lin.append(
                #nn.Sigmoid()
                nn.SELU()
                #nn.ReLU()
            )
            #self.stack.append(
            #    nn.Dropout(0)
            #)
        self.stack_lin.append(
            nn.Linear(self.sizes[-2], self.sizes[-1])
        )

    def forward(self, x):
        x = self.stack_conv(x)
        x = self.stack_trans(x)
        logits = self.stack_lin(x)
        #logits = self.stack(x)
        return logits

def create_dataset(
        x_train, y_train,
        x_valid, y_valid):

    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)
    return train_ds, valid_ds

def generate_data(
        train_data_size, validation_data_size,
        T_nmax, res,
        PATH_DATASET_TRAIN, PATH_DATASET_VALID,
        device):

    z = torch.zeros((train_data_size+validation_data_size, 1, res, res), device=device)
    y = torch.zeros((train_data_size+validation_data_size, T_nmax), device=device)

    one_idx = torch.randint(low=0, high=T_nmax, size=(train_data_size + validation_data_size,))
    for j in np.arange(train_data_size+validation_data_size):
        y[j, one_idx[j]] = 1

    r = torch.linspace(0, 1, res)
    for j in np.arange(train_data_size+validation_data_size):
        k = 2*np.pi*np.random.rand()*0.05
        t = torch.linspace(k, k+2*np.pi, res)
        R, T = torch.meshgrid(r, t)
        n_j, m_j = rc(one_idx[j]+1)
        z[j] = zernike_pol(n_j, m_j, R, T, cartesian=False)[n_j-1, m_j-1]
        #plt.imshow(z[j].reshape(res,res))
        #plt.show()
        #z[j].reshape(1,res,res)

    x_train = z[0:train_data_size]
    y_train = y[0:train_data_size]
    x_valid = z[train_data_size:-1]
    y_valid = y[train_data_size:-1]

    # create datasets
    train_ds, valid_ds = create_dataset(x_train, y_train, x_valid, y_valid)
    torch.save(train_ds, PATH_DATASET_TRAIN)
    torch.save(valid_ds, PATH_DATASET_VALID)
    return

def train(
        model,
        train_ds, valid_ds,
        optimizer, criterion,
        epochs, bs):

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs*2)
    for epoch in np.arange(epochs):
        model.train()
        for xb, yb in train_dl:
            pred = model(xb)
            #print(yb)
            #print(pred)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_loss = sum(criterion(net(xv), yv) for xv, yv in valid_dl)
        print("epoch: ", epoch, valid_loss / len(valid_dl))

    return

def comparison(model, T_nmax, res):
    k = 2*np.pi*np.random.rand()*0.01
    r = torch.linspace(0, 1, res)
    t = torch.linspace(k, k+2*np.pi, res)
    R, T = torch.meshgrid(r, t)

    y = torch.zeros((T_nmax,))
    idx = torch.randint(low=0, high=T_nmax, size=(1,))
    y[idx] = 1
    print(idx)

    n, m = rc(idx.item()+1)
    z = zernike_pol(n, m, R, T, cartesian=False)[n-1, m-1]\
    .unsqueeze(dim=0).unsqueeze(dim=0)

    pred = model(z)
    pred_prob = nn.functional.softmax(pred, dim=1)
    pred_idx = torch.argmax(pred_prob).item()
    n_p, m_p = rc(pred_idx+1)
    print(y)
    print(pred)
    print(pred_prob)
    print("Real orders: (",n,",",m,")")
    print("Predicted orders: (",n_p,",",m_p,")")
    print("Certainty: ", pred_prob[0,pred_idx].item())

    return

def save_model(model, SAVE_PATH):
    model_scripted = torch.jit.script(model)
    model_scripted.save(SAVE_PATH)
    return

def load_model(LOAD_PATH):
    model = torch.jit.load(LOAD_PATH)
    model.eval()
    return model

# check if gpu is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# paths
PATH_DIR_DATASETS = './saved/datasets/'
NAME_DATASET_TRAIN = 'dataset_train0.pt'
NAME_DATASET_VALID = 'dataset_valid0.pt'
PATH_DATASET_TRAIN = PATH_DIR_DATASETS+NAME_DATASET_TRAIN
PATH_DATASET_VALID = PATH_DIR_DATASETS+NAME_DATASET_VALID

PATH_DIR_MODELS = './saved/models/'
NAME_MODEL = 'model0.pt'
PATH_MODEL = PATH_DIR_MODELS+NAME_MODEL

# actions
create_new_data = True
perform_train = False
perform_comparison_test = True

# parameters
epochs = 100
train_data_size = 300
validation_data_size = 100
bs = int(train_data_size/10) # batch size
criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()
LR = 1e-3
MOM = 0.9
inner_lin_layers = [30, 24, 24, 21]
res = 224
n_max = 5
m_max = n_max
T_nmax = int((n_max+1)*(n_max+1+1)/2)

# generate data
if create_new_data:
    generate_data(train_data_size, validation_data_size,
                  T_nmax, res,
                  PATH_DATASET_TRAIN, PATH_DATASET_VALID,
                  device)

# load dataset
train_ds = torch.load(PATH_DATASET_TRAIN)
valid_ds = torch.load(PATH_DATASET_VALID)

# create linear layers
input_size = 100 # irrelevant for images
output_size = T_nmax

lin_layers = inner_lin_layers
lin_layers.append(output_size)
lin_layers.insert(0, input_size)

# create network instance
net = NeuralNetwork(lin_layers).to(device)

# create optimizer
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOM)

# train
if perform_train:
    train(net,
          train_ds, valid_ds,
          optimizer, criterion,
          epochs, bs)
    # save model
    save_model(net, PATH_MODEL)

#torch.save(net.state_dict(), SAVE_PATH)

# load parameters
eval_net = load_model(PATH_MODEL)
#eval_net = NeuralNetwork(layers)
#eval_net.load_state_dict(torch.load(SAVE_PATH))
#eval_net.eval()

# comparison
if perform_comparison_test:
    comparison(eval_net, T_nmax, res)
