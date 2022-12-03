import os
import io
import numpy as np
import torch
#import zernike
import torch.optim as optim
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split, Dataset
#from ray import tune
#from ray.tune import CLIReporter
#from ray.tune.schedulers import ASHAScheduler

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

import time
import h5py

class dataset_h5(Dataset):
    def __init__(self, in_file, transform=None, ds_type='train'):
        super(dataset_h5, self).__init__()
        self.ds_type=ds_type
        self.file = h5py.File(in_file, 'r')
        self.transform = transform

    def __getitem__(self, index):
        if (self.ds_type=='train'):
            x = self.file['X_train'][index, ...]
            y = self.file['Y_train'][index, ...]
        elif (self.ds_type=='valid'):
            x = self.file['X_test'][index, ...]
            y = self.file['Y_test'][index, ...]

        # Preprocessing each image
        if self.transform is not None:
            x = self.transform(x)

        #return (x, y), index
        return x, y

    def __len__(self):
        if (self.ds_type=='train'):
            return self.file['X_train'].shape[0]
        elif (self.ds_type=='valid'):
            return self.file['X_test'].shape[0]

def r_pol(n_max, Rho, device):
    mu_max = n_max
    #shape = Rho.shape
    R = torch.zeros((n_max+1, 2*n_max+1, Rho.size(0), Rho.size(1)), device=device)
    R[0,0] = torch.ones_like(Rho, device=device)
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

def radial_zpol(n_max, m_max, Rho, T, device):
    U = torch.zeros((n_max+1, m_max+1, Rho.size(0), Rho.size(1)), device=device)
    R = r_pol(n_max, Rho, device)
    for n_idx in np.arange(0, n_max+1):
        for m_idx in np.arange(0, m_max+1):
            mu_idx = n_idx - 2*m_idx
            if (mu_idx > 0):
                U[n_idx, m_idx,] = R[n_idx, np.abs(mu_idx)]*torch.sin(mu_idx*T)
            else:
                U[n_idx, m_idx,] = R[n_idx, np.abs(mu_idx)]*torch.cos(mu_idx*T)
    return U

def cartesian_zpol(n_max, m_max, X, Y, device):
    U = torch.zeros((n_max+1, m_max+1, X.size(0), X.size(1)), device=device)
    U[0,0,] = torch.zeros_like(X, device=device)
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

def zernike_pol(n,m, U, V, device, cartesian=True):
    if cartesian==True:
        S = cartesian_zpol(n, m, U, V, device)
    else:
        S = radial_zpol(n, m, U, V, device)
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

class NeuralNetwork(nn.Module):
    def __init__(self, sizes_conv, sizes_lin):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.sizes_lin = sizes_lin
        self.sizes_conv = sizes_conv
        self.num_layers_lin = len(sizes_lin)
        self.num_layers_conv = len(sizes_conv)
        self.stack_conv = nn.Sequential()
        self.stack_trans = nn.Sequential()
        self.stack_lin = nn.Sequential()

        self.conv_channels = sizes_conv

        if np.asarray(self.conv_channels).size != 0:
            #for i in np.arange(len(self.conv_channels)-1):
            #    self.stack_conv.append(
            #        nn.Conv2d(self.conv_channels[i],
            #                  self.conv_channels[i+1],
            #                  kernel_size=5, stride=1, padding='same')
            #    )
            #    self.stack_conv.append(
            #        nn.ReLU(inplace=True)
            #        #nn.SELU()
            #    )
            #    self.stack_conv.append(
            #        nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            #    )

            self.stack_conv = nn.Sequential(
                nn.Conv2d(2, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),)
            last_size = 6
            self.stack_trans.append(
                nn.AdaptiveAvgPool2d((last_size,last_size))
            )
            self.stack_trans.append(
                nn.Flatten(start_dim=1, end_dim=-1)
            )

            self.stack_lin.append(
                nn.Linear(self.conv_channels[-1]*last_size**2, self.sizes_lin[0])
            )
        else:
            self.stack_lin.append(
                nn.Flatten()
            )

       # self.stack.append(nn.Flatten())

        for i in np.arange(self.num_layers_lin-2):
            self.stack_lin.append(
                nn.Dropout()
            )
            self.stack_lin.append(
                nn.Linear(self.sizes_lin[i], self.sizes_lin[i+1])
            )
            self.stack_lin.append(
                #nn.Sigmoid()
                #nn.SELU()
                nn.ReLU(inplace=True)
            )
        self.stack_lin.append(
            nn.Linear(self.sizes_lin[-2], self.sizes_lin[-1])
        )

    def forward(self, x):
        x = self.stack_conv(x)
        x = self.stack_trans(x)
        logits = self.stack_lin(x)
        return logits

def write_dataset(
        x_train, y_train,
        x_valid, y_valid,
        PATH_DATASET):

    with h5py.File(PATH_DATASET, "w") as out:
        out.create_dataset("X_train",data=x_train.cpu())
        out.create_dataset("Y_train",data=y_train.cpu())
        out.create_dataset("X_test",data=x_valid.cpu())
        out.create_dataset("Y_test",data=y_valid.cpu())
    return

def generate_data(
        train_data_size, validation_data_size,
        T_nmax, res, n_max,
        PATH_DATASET,
        device):

    z = torch.zeros((train_data_size+validation_data_size, 2, res, res), device=device)
    y = torch.zeros((train_data_size+validation_data_size, 2*T_nmax), device=device)
    y = torch.rand(size=(train_data_size+validation_data_size, 2*T_nmax), device=device)*2-1

    #r = torch.linspace(0, 1, res, device=device)
    #t = torch.linspace(0, 2*np.pi, res, device=device)
    xs = torch.linspace(-1, 1, res, device=device)
    ys = torch.linspace(-1, 1, res, device=device)

    #R, T = torch.meshgrid(r, t)
    Xs, Ys = torch.meshgrid(xs, ys)
    n, m = rc(T_nmax)
    #U = zernike_pol(n, m, R, T, device, cartesian=False)
    U = zernike_pol(n, m, Xs, Ys, device)
    for j in np.arange(train_data_size+validation_data_size):
        S_real = torch.zeros((res, res), device=device)
        S_imag = torch.zeros((res, res), device=device)
        for i in np.arange(T_nmax):
            n_i, m_i = rc(i+1)
            S_real += y[j,i]*U[n_i-1, m_i-1]
            S_imag += y[j,T_nmax+i]*U[n_i-1, m_i-1]
        S_complex = torch.complex(S_real, S_imag)
        z[j,0] = S_real
        z[j,1] = S_imag
        #z[j,0] = torch.norm(S_complex)
        #z[j,1] = torch.angle(S_complex)

    x_train = z[0:train_data_size]
    y_train = y[0:train_data_size]
    x_valid = z[train_data_size:-1]
    y_valid = y[train_data_size:-1]

    # create datasets
    write_dataset(x_train, y_train, x_valid, y_valid,
        PATH_DATASET)
    print("Data has been generated...")
    return

def train(
        model,
        train_dl, valid_dl,
        optimizer, scheduler, criterion,
        epochs, bs, device):

    for epoch in np.arange(epochs):
        model.train()
        for xb, yb in train_dl:
            pred = model(xb.to(device))
            loss = criterion(pred, yb.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()

        model.eval()
        with torch.no_grad():
            #train_loss = sum(criterion(net(xb.to(device)), yb.to(device)) for xb, yb in train_dl)
            valid_loss = sum(criterion(net(xv.to(device)), yv.to(device)) for xv, yv in valid_dl)
        #print("epoch: ", epoch, "valid_loss: ",valid_loss / len(valid_dl), "train_loss: ", train_loss/len(train_dl))
        print("epoch: ", epoch, "valid_loss: ",valid_loss / len(valid_dl))
        #print("lr: ", scheduler.get_last_lr())

    return

def coefficientcomparison(T_nmax, C, C_pred):

    n_max, m_max = rc(T_nmax)

    # setup the figure and axes
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    x = torch.arange(n_max)
    y = torch.arange(n_max)
    X, Y = torch.meshgrid(x, y)
    xs = X.ravel()
    ys = Y.ravel()
    width=depth=1

    viridis = mpl.colormaps['viridis']

    top1 = torch.zeros((n_max, n_max))
    color1 = torch.zeros((n_max, n_max, 4))
    top2 = torch.zeros((n_max, n_max))
    color2 = torch.zeros((n_max, n_max, 4))

    counter = 0
    counter_img = T_nmax
    C_pred = torch.squeeze(C_pred, dim=0)

    Z = torch.complex(C[0:T_nmax], C[T_nmax:])
    Z_pred = torch.complex(C_pred[0:T_nmax], C[T_nmax:])

    for n in np.arange(n_max):
        for m in np.arange(n_max):
            if (m<=n):
                top1[n,m] = torch.norm(Z[counter])
                color1[n,m,:] = torch.tensor(viridis(torch.angle(Z[counter]).item()/(2*np.pi)))
                top2[n,m] = torch.norm(Z_pred[counter])
                color2[n,m,:] = torch.tensor(viridis(torch.angle(Z_pred[counter]).item()/(2*np.pi)))
                counter += 1
                counter_img += 1
            else:
                color1[n,m,:] = torch.tensor(viridis(0.5))
                color2[n,m,:] = torch.tensor(viridis(0.5))

    top = top1.ravel()
    bottom = torch.zeros_like(top)
    colors1 = torch.flatten(color1,start_dim=0, end_dim=-2).numpy()
    ax1.bar3d(xs, ys, bottom, width, depth, top, shade=True, color=colors1)
    ax1.set_title('Target')

    top = top2.ravel()
    bottom = torch.zeros_like(top)
    colors2 = torch.flatten(color2,start_dim=0, end_dim=-2).numpy()
    ax2.bar3d(xs, ys, bottom, width, depth, top, shade=True, color=colors2)
    ax2.set_title('Predicted')

    return

def comparison(model, T_nmax, res, device):
    n_max, m_max = rc(T_nmax)
    device = 'cpu'
    model.cpu()
    r = torch.linspace(0, 1, res, device=device)
    t = torch.linspace(0, 2*np.pi, res, device=device)
    R, T = torch.meshgrid(r, t)

    #xs = torch.linspace(-1, 1, res, device=device)
    #ys = torch.linspace(-1, 1, res, device=device)
    #Xs, Ys = torch.meshgrid(xs, ys)

    y = torch.rand((2*T_nmax,), device=device)*2-1

    U = zernike_pol(n_max, m_max, R, T, device, cartesian=False)
    #U = zernike_pol(n_max, m_max, Xs, Ys, device)
    #.unsqueeze(dim=0).unsqueeze(dim=0)

    Z = torch.zeros((1, 2, res, res), device=device)
    S_real = torch.zeros((res, res), device=device)
    S_imag = torch.zeros((res, res), device=device)
    for j in np.arange(T_nmax):
        n_j, m_j = rc(j+1)
        S_real += y[j]*U[n_j-1, m_j-1]
        S_imag += y[j+T_nmax]*U[n_j-1, m_j-1]
    Z[0,0] = S_real
    Z[0,1] = S_imag

    pred = model(Z)
    print("Coefficients: ", y)
    print("Predicted coefficients: ", pred)

    # calculate predicted Z
    Z_pred = torch.zeros(size=(2, res, res), device=device)
    S_pred_real = torch.zeros((res, res), device=device)
    S_pred_imag = torch.zeros((res, res), device=device)
    for j in np.arange(T_nmax):
        n_j, m_j = rc(j+1)
        S_pred_real += pred[0,j]*U[n_j-1, m_j-1]
        S_pred_imag += pred[0,j+T_nmax]*U[n_j-1, m_j-1]

    Z_pred[0] = S_pred_real
    Z_pred[1] = S_pred_imag

    Z.to('cpu')
    Z_pred.to('cpu')
    fig, ax = plt.subplots(subplot_kw = dict(projection='polar'))
    ax.pcolormesh(t, r, Z[0,0].cpu())
    plt.box(False)
    fig, ax = plt.subplots(subplot_kw = dict(projection='polar'))
    ax.pcolormesh(t, r, Z_pred[0].cpu())
    plt.box(False)
    fig, ax = plt.subplots(subplot_kw = dict(projection='polar'))
    ax.pcolormesh(t, r, Z[0,1].cpu())
    plt.box(False)
    fig, ax = plt.subplots(subplot_kw = dict(projection='polar'))
    ax.pcolormesh(t, r, Z_pred[1].cpu())
    plt.box(False)

    coefficientcomparison(T_nmax, y, pred)


    plt.show()

    return

def save_model(model, SAVE_PATH):
    model_scripted = torch.jit.script(model)
    model_scripted.save(SAVE_PATH)
    return

def load_model(LOAD_PATH, device):
    with open(LOAD_PATH, 'rb') as f:
        buffer = io.BytesIO(f.read())
    buffer.seek(0)
    model = torch.jit.load(LOAD_PATH, map_location=device)
    model.eval()
    return model

# check if gpu is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#torch.cuda.set_device(device)
device = torch.device(device)

# paths
PATH_DIR_DATASETS = './saved/datasets/'
NAME_DATASET = 'datasetjoined0.pt'
PATH_DATASET = PATH_DIR_DATASETS+NAME_DATASET

PATH_DIR_MODELS = './saved/models/'
NAME_MODEL = 'model_n6_0c.pt'
PATH_MODEL = PATH_DIR_MODELS+NAME_MODEL

# actions
create_new_data = False
perform_train = False
perform_comparison_test = True

# parameters
epochs = 1000
train_data_size = 2560*2
validation_data_size = 5000
bs = int(train_data_size/10) # batch size
#bs = 32
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss().to(device)
#LR = 1e-2
#LR = 1e-2
LR = 1e-5
MOM = 0.9
#conv_channels = [2, 64, 192, 384, 256, 256]
conv_channels = [2, 64, 256]

inner_lin_layers = [4096, 4096]
#conv_channels = []
#inner_lin_layers = [100, 140, 140, 110, 90, 10, 7]
res = 227
n_max = 6
m_max = n_max
T_nmax = int((n_max+1)*(n_max+1+1)/2)



# train
if perform_train:
    # generate data
    if create_new_data:
        generate_data(train_data_size, validation_data_size,
                      T_nmax, res, n_max,
                      PATH_DATASET,
                      device)

    # load dataset

    dataset_train = dataset_h5(PATH_DATASET, ds_type='train')
    dataset_valid = dataset_h5(PATH_DATASET, ds_type='valid')


    train_dl = torch.utils.data.DataLoader(
            dataset_train, batch_size=bs,
            shuffle=True)
    valid_dl = torch.utils.data.DataLoader(
            dataset_valid, batch_size=1,
            shuffle=False)

    # create linear layers
    input_size = 2*res**2 # irrelevant for images
    #input_size = 101
    output_size = 2*T_nmax

    lin_layers = inner_lin_layers
    lin_layers.append(output_size)
    lin_layers.insert(0, input_size)

    # create network instance
    net = NeuralNetwork(conv_channels, lin_layers).to(device)

    # create optimizer
    #optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOM)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    #scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.1)
    scheduler = 0

    train(net,
          train_dl, valid_dl,
          optimizer, scheduler, criterion,
          epochs, bs, device)

    # save model
    save_model(net, PATH_MODEL)

# load parameters
eval_net = load_model(PATH_MODEL, device)

# comparison
if perform_comparison_test:
    with torch.no_grad():
        comparison(eval_net, T_nmax, res, device)
