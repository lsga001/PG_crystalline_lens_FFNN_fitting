import os
import numpy as np
import scipy.special
import mpmath
import scipy.optimize
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
#import pychebfun

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import time
import h5py

from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

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
                #nn.ReLU()
                nn.SELU()
            )
            self.stack_lin.append(
                nn.Dropout(p=0.5)
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

def PG_fun(z, r, m, b, az, ar):
    return (b*z)**(m) * torch.exp(-b*z - (z/az)**2 - (r/ar)**2)

def contour_fun(z, x, h):
    return torch.abs(torch.log((x[1]*z)**x[0]/h) \
                     - x[1]*z - (z/x[2])**2)

def p_fun(q,a,b,az):
    #p = scipy.special.gamma(a)\
    #    *scipy.special.hyp1f1(a, q, (1/4) * az**2 * b**2)
    #print(p)

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
        #- 4*mu_z*(az/np.sqrt(2))*(m+1)\
        #* (scipy.special.pbdv(-(m+2), arg))\
        #/ (scipy.special.pbdv(-(m+1), arg))\
        #+ 2*mu_z**2
    sigma_z = torch.tensor(np.array(np.sqrt(sigma_z_sq).tolist(), dtype=float))
    mu_z = torch.tensor(np.array(mu_z.tolist(), dtype=float))
    return mu_z, sigma_z

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
        #print("m: ", m[j])
        #print("b: ", b[j])
        #print("az: ", az[j])
        #print("mu_z: ", mu_z[j])
        #print("delta: ", delta[j])
        #print("term_1 (+): ", term_1[j])
        #print("term_2 (-): ", term_2[j])
        #print("term_3 (+): ", term_3[j])
        #print("term_4 (-): ", term_4[j])
        #print("ssq: ", sigma_z_sq[j])
        #print("s: ", sigma_z[j])
        #if (torch.isnan(sigma_z[j])):
        #    print("is NAN")
        #    print("")
        #    print("")
        #    z = np.linspace(0,10, 1000)
        #    r = np.linspace(-5,5, 1000)
        #    Z,R = np.meshgrid(z, r)
        #    plt.imshow(PG_fun(Z, R, m[j], b[j], az[j], 3.523))
        #    plt.show()

    #print("ssq: ", torch.isnan(sigma_z_sq).any())
    #print("s: ", torch.isnan(sigma_z).any())
    return mu_z, sigma_z

def get_L0(x, h, z):
    m = x[:,0]
    b = x[:,1]
    az = x[:,2]
    return z*torch.log(1/h) \
        - m*z - b*z**2/2 \
        - z**3/(3*az**2) \
        + m*z*torch.log(b*z)

#def test_za_ar(x, mu_z, sigma_z, h, ze, zp):
#    z_zero = torch.zeros(x.size(dim=0))
#    v = pychebfun.chebfun('v', [0, 100])


#return za, ar

def get_za_and_ar(x, mu_z, sigma_z, h, ze, zp):
    #za = scipy.optimize.root(contour_fun, [0], args=(x,h))
    z_zero = torch.zeros(x.size(dim=0))
    for j in np.arange(x.size(dim=0)):
        print("j: ", j)

        #print(scipy.optimize.root_scalar(contour_fun, bracket=[0,zp[j]], args=(x[j], h[j])))
        #z_zero[j] = scipy.optimize.root_scalar(contour_fun, bracket=[0,zp[j]], args=(x[j], h[j])).root
        #z_zero[j],_,_ = scipy.optimize.fmin_l_bfgs_b(contour_fun, x0=ze[j], bounds=(0, zp[j]), args=(x[j], h[j]))

        #print("m: ", x[j,2])
        #print("b: ", x[j,3])
        #print("az: ", x[j,4])
        #print("V: ", x[j,5])
        #print("mu_z: ", mu_z[j])
        #print("sigma_z: ", sigma_z[j])
        #print("h: ", h[j])
        #print("zp: ", zp[j])
        z_zero[j] = scipy.optimize.fminbound(contour_fun, 0, ze[j].item(), args=(x[j], h[j]))
        #print(z_zero[j])
    za = ze - z_zero
    d = za + zp
    print("z_a: ", za)
    print("z_p: ", zp)
    print("d: ", d)
    L0 = get_L0(x,h,mu_z+sigma_z) - get_L0(x,h,mu_z+sigma_z-d)
    return za, torch.sqrt(x[:,3]/(np.pi*L0))

def R(b, m, az, ar, z):
    return (ar**2)/2 * torch.abs(-b + m/z - 2*z/(az**2))

def generate_td(x):
    m = x[:,0]
    b = x[:,1]
    az = x[:,2]
    V = x[:,3]

    #mu_z, sigma_z = get_sigma_z_and_mu_z(m, b, az)
    mu_z, sigma_z = test_m_sig(m, b, az)

    ze = (1/4)*(-az**2 * b + torch.sqrt(az**4 * b**2 + 8*az**2 * m))
    zp = mu_z + sigma_z - ze

    l = PG_fun(ze, 0, m,b,az,1)
    h = PG_fun(mu_z+sigma_z, 0, m,b,az,1)

    za, ar = get_za_and_ar(x, mu_z, sigma_z, h, ze, zp)

    Ra = R(b, m, az, ar, ze - za)
    Rp = R(b, m, az, ar, ze + zp)
    d = za + zp
    Re = ar * torch.sqrt(torch.log((b*ze)**m / h) - b*ze - (ze/az)**2)

    out = torch.zeros_like(x)
    out[:,0] = Ra
    out[:,1] = Rp
    out[:,2] = d
    out[:,3] = Re

    return out

def generate_data(lims,
        td_size, vd_size,
        PATH_DATASET,
        device):

    x = torch.zeros(td_size+vd_size, 4)#.to(device)
    y = torch.zeros(td_size+vd_size, 4)
    for j in np.arange(0, x.size(dim=1)):
        x[:,j] = lims[j,0] + torch.rand(td_size+vd_size)*(lims[j,1] - lims[j,0]) # low + k*(high-low)

    # get target output
    y = generate_td(x)
    print("x", x.size(dim=0))
    print("y", y.size(dim=0))
    nan_idx = np.isnan(y.numpy())
    nan_idx = np.any(nan_idx, axis=1)
    print("nans", nan_idx)

    x_train = torch.Tensor(np.delete(y[0:td_size].numpy(), nan_idx[0:td_size], 0))
    y_train = torch.Tensor(np.delete(x[0:td_size].numpy(), nan_idx[0:td_size], 0))
    x_valid = torch.Tensor(np.delete(y[td_size:-1].numpy(), nan_idx[td_size:-1], 0))
    y_valid = torch.Tensor(np.delete(x[td_size:-1].numpy(), nan_idx[td_size:-1], 0))

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

def load_data(PATH_DATASET):
    ds_train = dataset_h5(PATH_DATASET, ds_type='train')
    ds_valid = dataset_h5(PATH_DATASET, ds_type='valid')
    return ds_train, ds_valid

def train(config, checkpoint_dir=None, PATH_DATASET=None):
    # check if gpu is available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    device = torch.device(device)

    model = NeuralNetwork(sizes_conv=config["layers_conv"],
                sizes_lin=config["layers_lin"]).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr = config["lr"], momentum=config["mom"])

    # load dataset

    dataset_train, dataset_valid = load_data(PATH_DATASET)

    train_dl = torch.utils.data.DataLoader(
            dataset_train, batch_size=config["bs"],
            shuffle=True)
    valid_dl = torch.utils.data.DataLoader(
            dataset_valid,
            shuffle=False)

    for epoch in np.arange(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_loss = sum(criterion(model(xv), yv) for xv, yv in valid_dl)
            train_loss = sum(criterion(model(xb), yb) for xb, yb, in train_dl)
        print("epoch: ", epoch, "valid_loss: ",valid_loss / len(valid_dl), "train_loss: ",train_loss/len(train_dl))
        #print("epoch: ", epoch, "valid_loss: ",valid_loss / len(valid_dl))
        #scheduler.step()
        tune.report(loss=valid_loss)
    return

def comparison(model, device):
    physical_params = torch.Tensor([12.94, 5.988, 3.58, 3.76]).to(device)
    params = model(physical_params).cpu().unsqueeze(dim=0)
    model_params = generate_td(params)
    print("input physical parameters: ", physical_params)
    print("predicted params: ", params)
    print("calculated physical parameters from predicted params: ", model_params)
    print("")

    physical_params = torch.Tensor([12.05, 5.82, 3.62, 3.76]).to(device)
    params = model(physical_params).cpu().unsqueeze(dim=0)
    model_params = generate_td(params)
    print("input physical parameters: ", physical_params)
    print("predicted params: ", params)
    print("calculated physical parameters from predicted params: ", model_params)
    print("")
    return

def save_model(model, SAVE_PATH):
    model_scripted = torch.jit.script(model)
    model_scripted.save(SAVE_PATH)
    return

def load_model(LOAD_PATH, device):
    model = torch.jit.load(LOAD_PATH, map_location=device)
    model.eval()
    return model

# check if gpu is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

device = torch.device(device)

# paths
PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PATH_DIR_DATASETS = PATH_ROOT_DIR + '/saved/datasets/'

NAME_DATASET = 'ds4.pt'
PATH_DATASET = PATH_DIR_DATASETS+NAME_DATASET

PATH_DIR_MODELS = PATH_ROOT_DIR + '/saved/models/'
NAME_MODEL = 'm4_2.pt'
PATH_MODEL = PATH_DIR_MODELS+NAME_MODEL

# actions
create_new_data = False
perform_train = True
perform_comparison_test = True

# parameters

lims = np.zeros((4,2))

lims[0,0], lims[0,1] = 0, 18 #m_min, m_max;
lims[1,0], lims[1,1] = 0.00, 2.00 #b_min, b_max;
lims[2,0], lims[2,1] = 2.00, 5.00 #az_min, az_max;
#lims[3,0], lims[3,1] = 0, 10 #ar_min, ar_max;
lims[3,0], lims[3,1] = 80, 200 #V_min, V_max; V>=0

test=torch.zeros((1,4))
test[0,0] = 2.8 # m
test[0,1] = 0.67 # b
test[0,2] = 3.15 # az
test[0,3] = 106.5 #V
generate_td(test)
#time.sleep(100)

epochs = 5
train_data_size = 50000
validation_data_size = 20000
#bs = int(train_data_size/1)
bs = 32
criterion = nn.MSELoss()
LR = 3.3e-3
MOM = 0.9
weight_decay = 1e-5
conv_channels = []
inner_lin_layers = [1024, 512, 1024, 256]
#inner_lin_layers = [1024, 1024]
#inner_lin_layers = [5, 10, 9, 6]
#inner_lin_layers = [200, 50, 20, 10]

config = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "mom": tune.uniform(0.1, 0.9),
    "layers_conv": [],
    "layers_lin": [4, 1024, 512, 1024, 256, 4],
    "bs": bs,
}

#res = 227

# generate data
if create_new_data:
    generate_data(lims,
        train_data_size, validation_data_size,
        PATH_DATASET,
        device)


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
#optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay)
#optimizer = optim.Adamax(net.parameters(), lr=LR)
#scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.1)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
#scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-2, total_iters=epochs)

scheduler = ASHAScheduler(
    metric = "loss",
    mode = "min",
    max_t = epochs,
    grace_period = 1,
    reduction_factor = 2
)

reporter = CLIReporter(
    metric_columns=["loss", "training_iteration"]
)

# train
if perform_train:
    result = tune.run(
        partial(train, PATH_DATASET=PATH_DATASET),
        config=config,
        num_samples=100,
        scheduler=scheduler,
        progress_reporter=reporter
    )
    #train(config, PATH_DATASET=PATH_DATASET)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    best_trained_model = NeuralNetwork(best_trial.config["layers_conv"],
                             best_trial.config["layers_lin"])
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)
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
