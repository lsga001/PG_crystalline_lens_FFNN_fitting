import os
import numpy as np
import scipy.special
import mpmath
import scipy.optimize
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import time
import h5py

class dataset_h5(torch.utils.data.Dataset):
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
            x, y = self.transform(x, y)

        #return (x, y), index
        return x, y

    def __len__(self):
        if (self.ds_type=='train'):
            return self.file['X_train'].shape[0]
        elif (self.ds_type=='valid'):
            return self.file['X_test'].shape[0]

def g_fun(a, x, z):
    return x*torch.exp(a*x**2 + x) - z

def g_p_fun(a, x):
    return (2*a*x**2 + x + 1)*torch.exp(a*x**2+x)

def g_pp_fun(a, x):
    return (4 * a**2 * x**3 + 4*a*x**2 + (6*a+1)*x + 2)*torch.exp(a*x**2 + x)

def gpp_gp_fun(a, x):
    return (4*a*x+1)/(2*a*x**2+x+1) + 2*a*x + 1

def gp_g_fun(a, x, z):
    return x/(2*a*x**2+x+1) - z*(2*a*x**2+x+1)*torch.exp(-a*x**2 - x)

def qW_fun(a, z):
    S = torch.zeros_like(z)
    n_iters=18
    for k in np.arange(0, n_iters):
        g = g_fun(a, S, z)
        g_p = g_p_fun(a, S)
        g_pp = g_pp_fun(a, S)
        S = S - (2*g*g_p)/(2*g_p**2 - g*g_pp)

        #gp_g = gp_g_fun(a, S, z)
        #gpp_gp = gpp_gp_fun(a, S)
        #S = S - gp_g / (1 - gp_g*gpp_gp)
    return S

def qW_p_fun(a, z):
    qW = qW_fun(a, z)
    return 1/(torch.exp(a*qW**2 + qW)*(1 + 2*a*qW**2 + qW))

#class qW_torch(torch.autograd.Function):


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
        #else:
            #self.stack_lin.append(
            #    nn.Flatten()
            #)

        for i in np.arange(self.num_layers_lin-2):
            self.stack_lin.append(
                nn.Linear(self.sizes_lin[i], self.sizes_lin[i+1])
            )
            #self.stack_lin.append(
                #nn.BatchNorm1d(self.sizes_lin[i+1])
                #nn.LayerNorm(self.sizes_lin[i])
            #)
            self.stack_lin.append(
                nn.SELU()
            )
            #self.stack_lin.append(
            #    nn.Dropout(p=0.25)
            #)
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

def PG_fun(z, r, m, b, az, ar):
    return (b*z)**(m) * torch.exp(-b*z - (z/az)**2 - (r/ar)**2)

def contour_fun(z, x, h):
    return torch.abs(torch.log((x[1]*z)**x[0]/h) \
                     - x[1]*z - (z/x[2])**2)

def p_fun(q,a,b,az):

    a = a.numpy()
    b = b.numpy()
    az = az.numpy()

    my_gamma = np.frompyfunc(mpmath.gamma, 1, 1)
    my_hyp1f1 = np.frompyfunc(mpmath.hyp1f1, 3, 1)
    p = my_gamma(a)\
        *my_hyp1f1(a, q, (1/4) * (az*b)**2)
    #print(p)
    return torch.tensor(np.array(p.tolist(), dtype=float))

def get_delta(m,b,az):
    delta = az*p_fun(1/2,(m+1)/2,b,az) \
        - az**2 * b * p_fun(3/2,m/2+1,b,az)
    return delta

def test_m_sig(m, b, az):
    m = m.numpy()
    b = b.numpy()
    az = az.numpy()

    arg = b*az/np.sqrt(2)
    mu_z,_ = az/np.sqrt(2) \
        * (m+1) \
        * (scipy.special.pbdv(-(m+2), arg))\
        / (scipy.special.pbdv(-(m+1), arg))

    sigma_z_sq,_ = az**2 * (m+2) * (m+1) \
        * (scipy.special.pbdv(-(m+3), arg))\
        / (scipy.special.pbdv(-(m+1), arg))\
        - 2*mu_z**2
    sigma_z = torch.tensor(np.array(np.sqrt(sigma_z_sq).tolist(), dtype=float))
    mu_z = torch.tensor(np.array(mu_z.tolist(), dtype=float))
    return mu_z, sigma_z
    #return mu_z, torch.sqrt(sigma_z_sq)

def get_sigma_z_and_mu_z(m,b,az):
    mu_z = (p_fun(1/2,(m/2)+1, b, az) - az*b*p_fun(3/2,(m+3)/2,b,az)) \
        / ((1/az)*p_fun(1/2,(m+1)/2,b,az) - b*p_fun(3/2,(m/2)+1,b,az))
    test_mu, test_sigma = test_m_sig(m, b, az)
    print("mu_z before: ", mu_z)
    print("mu_z after: ", test_mu)
    delta = get_delta(m,b,az)
    #print("delta: ", torch.isnan(delta).any())
    term_1 = (az**2/delta)* (az*b*m/2)*(az**2 * b + 4*mu_z)*p_fun(3/2,(m+1)/2,b,az)
    term_2 = -(az**2/delta) * b*(az**2 * (m+1) + 2*mu_z**2)*p_fun(3/2,(m/2)+1,b,az)
    term_3 = (az**2/delta)*(az/2)*(2*(m+1) + (az*b + (2*mu_z/az))**2)*p_fun(1/2,(m+1)/2,b,az)
    term_4 = -(az**2/delta)*(az**2 * b + 4*mu_z) * p_fun(1/2,(m/2)+1,b,az)
    sigma_z_sq = ((az**2)/delta) \
        * (\
           (az*b*m/2)*(az**2 * b + 4*mu_z)*p_fun(3/2,(m+1)/2,b,az)\
           -b*(az**2 * (m+1) + 2*mu_z**2)*p_fun(3/2,(m/2)+1,b,az)\
           +(az/2)*(2*(m+1) + (az*b + (2*mu_z/az))**2)*p_fun(1/2,(m+1)/2,b,az)\
           -(az**2 * b + 4*mu_z) * p_fun(1/2,(m/2)+1,b,az)\
           )

    sigma_z = torch.sqrt(sigma_z_sq)
    print("sigma_z before: ", sigma_z)
    print("sigma_z after: ", test_sigma)
    for j in np.arange(sigma_z.size(dim=0)):
        print(j)
    return mu_z, sigma_z

def get_L0(x, h, z):
    m = x[:,0]
    b = x[:,1]
    az = x[:,2]
    return z*torch.log(1/h) \
        - m*z - b*z**2/2 \
        - z**3/(3*az**2) \
        + m*z*torch.log(b*z)

def R(b, m, az, ar, z):
    return (ar**2)/2 * torch.abs(-b + m/z - 2*z/(az**2))

def generate_td(x):
    m = x[:,0]
    b = x[:,1]
    az = x[:,2]
    #V = x[:,3]
    ar = x[:,3]

    t0 = time.time()
    #mu_z, sigma_z = get_sigma_z_and_mu_z(m, b, az)
    mu_z, sigma_z = test_m_sig(m, b, az)

    t1 = time.time()
    print("time in mu and sig: ", t1 - t0)

    ze = (1/4)*(-az**2 * b + torch.sqrt(az**4 * b**2 + 8*az**2 * m))
    zp = mu_z + sigma_z - ze

    l = PG_fun(ze, 0, m,b,az,1)
    h = PG_fun(mu_z+sigma_z, 0, m,b,az,1)

    t1 = time.time()

    cut_a = -(m/b)*qW_fun(-m / (az**2 * b**2), - h**(1/m) / m)
    t2 = time.time()
    za = ze - cut_a
    print("time taken za: ", t2-t1)

    Ra = R(b, m, az, ar, ze - za)
    Rp = R(b, m, az, ar, ze + zp)
    d = za + zp
    #Re = ar * torch.sqrt(torch.log((b*ze)**m / h) - b*ze - (ze/az)**2)

    L0 = get_L0(x,h,mu_z+sigma_z) - get_L0(x,h,mu_z+sigma_z-d)
    V = ar**2 * np.pi * L0
    t3 = time.time()
    print("time taken after: ", t3-t2)

    print("Ra: ", Ra)
    print("Rp: ", Rp)
    print("d: ", d)
    #print("Re: ", Re)
    print("V: ", V)

    out = torch.zeros_like(x)
    out[:,0] = Ra
    out[:,1] = Rp
    out[:,2] = d
    #out[:,3] = Re
    out[:,3] = V

    return out

def generate_data(lims,
        td_size, vd_size,
        PATH_DATASET,
        device):

    x = torch.zeros(td_size+vd_size, 4)#.to(device)
    y = torch.zeros(td_size+vd_size, 4)#.to(device)
    for j in np.arange(0, x.size(dim=1)):
        x[:,j] = lims[j,0] + torch.rand(td_size+vd_size)*(lims[j,1] - lims[j,0]) # low + k*(high-low)

    # get target output
    y = generate_td(x)
    print("x", x.size(dim=0))
    print("y", y.size(dim=0))
    print("x", x)
    print("y", y)
    nan_idx = np.isnan(y.numpy())
    nan_idx = np.any(nan_idx, axis=1)
    print("nans", nan_idx)

    y = y.numpy()
    # Remove outliers
    outliers_0 = (y[:,0] < lims[4,0]) | (y[:,0] > lims[4,1])
    outliers_1 = (y[:,1] < lims[5,0]) | (y[:,1] > lims[5,1])
    outliers_2 = (y[:,2] < lims[6,0]) | (y[:,2] > lims[6,1])
    outliers_3 = (y[:,3] < lims[7,0]) | (y[:,3] > lims[7,1])
    outliers = outliers_0 | outliers_1 | outliers_2 | outliers_3
    print("outliers shape", outliers.shape)
    outliers_idx = outliers
    print("outliers Ra:", outliers_0)
    print("outliers Rp:", outliers_1)
    print("outliers d:", outliers_2)
    #print("outliers Re:", outliers_3)
    print("outliers V:", outliers_3)
    print("outliers", outliers_idx)

    y = torch.Tensor(y)
    remove_idx = nan_idx | outliers_idx
    # remove data
    x_train = torch.Tensor(np.delete(y[0:td_size].numpy(), remove_idx[0:td_size], 0))
    y_train = torch.Tensor(np.delete(x[0:td_size].numpy(), remove_idx[0:td_size], 0))
    x_valid = torch.Tensor(np.delete(y[td_size:-1].numpy(), remove_idx[td_size:-1], 0))
    y_valid = torch.Tensor(np.delete(x[td_size:-1].numpy(), remove_idx[td_size:-1], 0))

    # scale volume
    #y_train[:,3] = y_train[:,3] / 100
    #y_valid[:,3] = y_valid[:,3] / 100

    print("x_train: ", x_train.size(0))
    print("y_train: ", y_train.size(0))
    print("x_valid: ", x_valid.size(0))
    print("y_valid: ", y_valid.size(0))

    print("x_train nan: ", x_train.isnan().any())
    print("y_train nan: ", y_train.isnan().any())
    print("x_valid nan: ", x_valid.isnan().any())
    print("y_valid nan: ", y_valid.isnan().any())

    print("x_train", x_train)
    print("y_train", y_train)
    print("x_valid", x_valid)
    print("y_valid", y_valid)
    # create datasets
    write_dataset(x_train, y_train, x_valid, y_valid,
        PATH_DATASET)
    print("Data has been generated...")
    return

def transform_error(x):
    TE = x
    Err_max = [0.1, 0.1, 0.1, 5] # Ra, Rp, d, V

    TE[:,0] = torch.exp( (np.log(2) / Err_max[0]) * x[:,0]) - 1
    TE[:,1] = torch.exp( (np.log(2) / Err_max[1]) * x[:,1]) - 1
    TE[:,2] = torch.exp( (np.log(2) / Err_max[2]) * x[:,2]) - 1
    TE[:,3] = torch.exp( (np.log(2) / Err_max[3]) * x[:,3]) - 1
    return TE

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

            #some_error = torch.abs(generate_td(pred) - xb.to(device))
            #print("some_error", some_error)
            #transformed_error = transform_error(some_error)
            #print("Transformed error: ", transformed_error)
            #loss = criterion(transformed_error, some_error*0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            valid_loss = sum(criterion(model(xv.to(device)), yv.to(device)) for xv, yv in valid_dl)
            train_loss = sum(criterion(model(xb.to(device)), yb.to(device)) for xb, yb, in train_dl)
        print("epoch: ", epoch, "valid_loss: ",valid_loss / len(valid_dl), "train_loss: ",train_loss/len(train_dl))
        #print("epoch: ", epoch, "valid_loss: ",valid_loss / len(valid_dl))

    return

def comparison(model, device):
    physical_params = torch.Tensor([12.94, 5.988, 3.58, 106.5]).to(device)
    params = model(physical_params).cpu().unsqueeze(dim=0)
    model_params = generate_td(params)
    print("input physical parameters: ", physical_params)
    print("predicted params: ", params)
    print("calculated physical parameters from predicted params: ", model_params)
    print("")

    physical_params = torch.Tensor([12.05, 5.82, 3.62, 106.5]).to(device)
    params = model(physical_params).cpu().unsqueeze(dim=0)
    model_params = generate_td(params)
    print("input physical parameters: ", physical_params)
    print("predicted params: ", params)
    print("calculated physical parameters from predicted params: ", model_params)
    print("")

    physical_params = torch.Tensor([8, 5.5, 4.2, 106.5]).to(device)
    params = model(physical_params).cpu().unsqueeze(dim=0)
    model_params = generate_td(params)
    print("input physical parameters: ", physical_params)
    print("predicted params: ", params)
    print("calculated physical parameters from predicted params: ", model_params)
    print("")

    num = 1000
    t = torch.linspace(0, 56, num) # years old
    Ra = 11.453*torch.exp(-torch.exp(-0.0566*t))
    Rp = 6.47*torch.exp(-torch.exp(-0.0667*t - 0.457))
    d = -1.32*1e-5 * t**3 + 0.00182*t**2 - 0.0658*t + 4.802
    V = 3.166 * t**(0.832) + 110.4

    input_prop = torch.zeros(num, 4)
    predicted_params = torch.zeros(num, 4)
    output_prop = torch.zeros(num, 4)

    input_prop[:,0] = Ra
    input_prop[:,1] = Rp
    input_prop[:,2] = d
    input_prop[:,3] = V

    predicted_params = model(input_prop).cpu()
    output_prop = generate_td(predicted_params)

    for i in np.arange(num):
        print()
        print("Given: ", input_prop[i])
        print("Recieved: ", output_prop[i])
        print()

    plt.figure(0)
    plt.suptitle("Geometric properties of the crystalline lens with respect to age")
    ax0 = plt.subplot(221)
    ax0.plot(t, input_prop[:,0], label='desired')
    ax0.plot(t, output_prop[:,0], label='generated')
    ax0.set_xlabel("Age [y]")
    ax0.set_ylabel("RAL [mm]")
    ax0.legend()

    ax1 = plt.subplot(222)
    ax1.plot(t, input_prop[:,1], label='desired')
    ax1.plot(t, output_prop[:,1], label='generated')
    ax1.set_xlabel("Age [y]")
    ax1.set_ylabel("RPL [mm]")
    ax1.legend()

    ax2 = plt.subplot(223)
    ax2.plot(t, input_prop[:,2], label='desired')
    ax2.plot(t, output_prop[:,2], label='generated')
    ax2.set_xlabel("Age [y]")
    ax2.set_ylabel("LT [mm]")
    ax2.legend()

    ax3 = plt.subplot(224)
    ax3.plot(t, input_prop[:,3], label='desired')
    ax3.plot(t, output_prop[:,3], label='generated')
    ax3.set_xlabel("Age [y]")
    ax3.set_ylabel("VOL [mm^3]")
    ax3.legend()

    plt.figure(1)
    plt.suptitle("Generated model relative error with respect to age")
    ax0 = plt.subplot(221)
    ax0.plot(t, 100*torch.abs(input_prop[:,0] - output_prop[:,0])/input_prop[:,0])
    ax0.set_xlabel("Age [y]")
    ax0.set_ylabel("Err(RAL) [%]")

    ax1 = plt.subplot(222)
    ax1.plot(t, 100*torch.abs(input_prop[:,1] - output_prop[:,1])/input_prop[:,1])
    ax1.set_xlabel("Age [y]")
    ax1.set_ylabel("Err(RPL) [%]")

    ax2 = plt.subplot(223)
    ax2.plot(t, 100*torch.abs(input_prop[:,2] - output_prop[:,2])/input_prop[:,2])
    ax2.set_xlabel("Age [y]")
    ax2.set_ylabel("Err(LT) [%]")

    ax3 = plt.subplot(224)
    ax3.plot(t, 100*torch.abs(input_prop[:,3] - output_prop[:,3])/input_prop[:,3])
    ax3.set_xlabel("Age [y]")
    ax3.set_ylabel("Err(VOL) [%]")

    plt.show()
    return

def save_model(model, SAVE_PATH):
    model_scripted = torch.jit.script(model)
    model_scripted.save(SAVE_PATH)
    return

def load_model(LOAD_PATH, device):
    model = torch.jit.load(LOAD_PATH, map_location=device)
    model.eval()
    return model

def desirable_parameters_generation(model):
    # Ra, Rp, d, V; in [mm]
    age = 20
    Data = [
        [5.81+0.085*age, 5.33+0.010*age, 3.54+0.019*age, 105.28+1.766*age], # European
        [7.25+0.072*age, 5.39+0.018*age, 3.67+0.015*age, 112.59+1.666*age], # Indian
        [7.14, 5.57, 4.62, 178], # S1 (OD)
        [8.94, 5.80, 4.35, 177], # S2 (OS)
        [7.77, 5.46, 5.67, 264], # S3 (OD)
        [7.62, 5.05, 5.61, 246], # S3 (OS)
    ]
    Data = torch.Tensor(Data)
    Params = model(Data)
    Properties = generate_td(Params)
    print("Data: ", Data)
    print("Params: ", Params)
    print("Properties: ", Properties)
    print("Rel Err [%]: ", 100*torch.abs(Data-Properties)/Data)
    return

# check if gpu is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

device = torch.device(device)

# paths
PATH_DIR_DATASETS = './saved/datasets/'

NAME_DATASET = 'ds7.pt'
PATH_DATASET = PATH_DIR_DATASETS+NAME_DATASET

PATH_DIR_MODELS = './saved/models/'
NAME_MODEL = 'm7_0.pt'
PATH_MODEL = PATH_DIR_MODELS+NAME_MODEL

# actions
create_new_data = False
perform_train = False
perform_comparison_test = True
generate_desirable_parameters = True

# parameters

lims = np.zeros((8,2))

lims[0,0], lims[0,1] = 0, 20 #m_min, m_max;
lims[1,0], lims[1,1] = 0.50, 1.00 #b_min, b_max;
lims[2,0], lims[2,1] = 2.5, 3.5 #az_min, az_max;
lims[3,0], lims[3,1] = 2, 5 #ar_min, ar_max;
#lims[3,0], lims[3,1] = 100, 250 #V_min, V_max; V>=0

lims[4,0], lims[4,1] = 1, 15 # Ra_min, Ra_max
lims[5,0], lims[5,1] = 2, 10 # Rp_min, Rp_max
lims[6,0], lims[6,1] = 2, 7 # d_min, d_max
#lims[7,0], lims[7,1] = 1, 7 # Re_min, Re_max
lims[7,0], lims[7,1] = 100, 300 # V_min, V_max

test=torch.zeros((1,4))
test[0,0] = 2.8 # m
test[0,1] = 0.67 # b
test[0,2] = 3.15 # az
#test[0,3] = 106.5 #V
test[0,3] = 3.523# ar
generate_td(test)

epochs = 30
train_data_size = 500000
validation_data_size = 250000
#bs = int(train_data_size/100)
bs = 2048
criterion = nn.MSELoss()
#criterion = nn.KLDivLoss(reduction='batchmean')
#criterion = MeanAbsoluteRelativeError()
LR = 1e-4
MOM = 0.4
#weight_decay = 1e-5
weight_decay = 0
conv_channels = []
inner_lin_layers = [512, 2048, 2048, 512, 256]
#inner_lin_layers = [1024]
#inner_lin_layers = [1024, 1024]
#inner_lin_layers = [5, 10, 9, 6]
#inner_lin_layers = [200, 50, 20, 10]

#res = 227

# generate data
if create_new_data:
    generate_data(lims,
        train_data_size, validation_data_size,
        PATH_DATASET,
        device)

# load dataset
def my_transform(x, y):
    #y[3] = y[3]#/100
    x = x #/ 25
    y = y
    return x, y

dataset_train = dataset_h5(PATH_DATASET, ds_type='train', transform=my_transform)
dataset_valid = dataset_h5(PATH_DATASET, ds_type='valid', transform=my_transform)

train_dl = torch.utils.data.DataLoader(
        dataset_train, batch_size=bs,
        shuffle=True)
valid_dl = torch.utils.data.DataLoader(
        dataset_valid, batch_size=validation_data_size,
        shuffle=False)

#for xb, yb in train_dl:
#    print("Max train", torch.max(xb), torch.max(yb))
#for xv, yv in valid_dl:
#    print("Max valid", torch.max(xv), torch.max(yv))

# create linear layers
input_size = 4
output_size = 4

lin_layers = inner_lin_layers
lin_layers.append(output_size)
lin_layers.insert(0, input_size)

# create network instance
net = NeuralNetwork(conv_channels, lin_layers).to(device)

# create optimizer
#optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOM)
optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay)
#optimizer = optim.Adamax(net.parameters(), lr=LR)
#scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, cycle_momentum=False, step_size_up=100)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
#scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-2, total_iters=epochs)

# train
if perform_train:
    train(net,
          train_dl, valid_dl,
          optimizer, scheduler, criterion,
          epochs, bs, device)
    # save model
    save_model(net, PATH_MODEL)

#torch.save(net.state_dict(), SAVE_PATH)

# comparison
if perform_comparison_test:
    # load parameters
    eval_net = load_model(PATH_MODEL, device)
    #eval_net = NeuralNetwork(layers)
    #eval_net.load_state_dict(torch.load(SAVE_PATH))
    #eval_net.eval()

    with torch.no_grad():
        comparison(eval_net, device)

if generate_desirable_parameters:
    eval_net = load_model(PATH_MODEL, device)
    with torch.no_grad():
        desirable_parameters_generation(eval_net)
