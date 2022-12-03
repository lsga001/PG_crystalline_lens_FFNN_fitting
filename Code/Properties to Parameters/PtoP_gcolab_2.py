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
            
            #self.stack_lin.append(
                #nn.BatchNorm1d(self.sizes_lin[i])
                #nn.LayerNorm(self.sizes_lin[i])
            #)
            self.stack_lin.append(
                nn.Linear(self.sizes_lin[i], self.sizes_lin[i+1])
            )
            self.stack_lin.append(
                nn.ReLU()
            )
            #self.stack_lin.append(
            #    nn.Dropout(p=0.4)
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

def get_sigma_z_and_mu_z(m,b,az):
    mu_z = (p_fun(1/2,(m/2)+1, b, az) - az*b*p_fun(3/2,(m+3)/2,b,az)) \
        / ((1/az)*p_fun(1/2,(m+1)/2,b,az) - b*p_fun(3/2,(m/2)+1,b,az))
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

def get_za_and_ar(x, mu_z, sigma_z, h, ze, zp):
    #za = scipy.optimize.root(contour_fun, [0], args=(x,h))
    z_zero = torch.zeros(x.size(dim=0))
    for j in np.arange(x.size(dim=0)):
        print(j)

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

    mu_z, sigma_z = get_sigma_z_and_mu_z(m, b, az)

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

        model.eval()
        with torch.no_grad():
            valid_loss = sum(criterion(net(xv.to(device)), yv.to(device)) for xv, yv in valid_dl)
            train_loss = sum(criterion(net(xb.to(device)), yb.to(device)) for xb, yb, in train_dl)
        print("epoch: ", epoch, "valid_loss: ",valid_loss / len(valid_dl), "train_loss: ",train_loss/len(train_dl))
        scheduler.step()

    return

def comparison(model, device):
    return

def save_model(model, SAVE_PATH):
    model_scripted = torch.jit.script(model)
    model_scripted.save(SAVE_PATH)
    return

def load_model(LOAD_PATH, device):
    model = torch.jit.load(LOAD_PATH).to(device)
    model.eval()
    return model

# check if gpu is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#torch.cuda.set_device(device)
device = torch.device(device)

# paths
PATH_DIR_DATASETS = './saved/datasets/'

NAME_DATASET = 'ds1.pt'
PATH_DATASET = PATH_DIR_DATASETS+NAME_DATASET

PATH_DIR_MODELS = './saved/models/'
NAME_MODEL = 'm1_1.pt'
PATH_MODEL = PATH_DIR_MODELS+NAME_MODEL

# actions
create_new_data = False
perform_train = True
perform_comparison_test = False

# parameters

lims = np.zeros((4,2))

lims[0,0], lims[0,1] = 2, 10 #m_min, m_max;
lims[1,0], lims[1,1] = 0.60, 0.80 #b_min, b_max;
lims[2,0], lims[2,1] = 3.10, 3.20 #az_min, az_max;
#lims[3,0], lims[3,1] = 0, 10 #ar_min, ar_max;
lims[3,0], lims[3,1] = 100, 110 #V_min, V_max; V>=0

test=torch.zeros((1,4))
test[0,0] = 2.8 # m
test[0,1] = 0.67 # b
test[0,2] = 3.15 # az
test[0,3] = 106.5 #V
generate_td(test)
#time.sleep(100)

epochs = 50
train_data_size = 50000
validation_data_size = 20000
#bs = int(train_data_size/10)
bs = 32
criterion = nn.MSELoss()
LR = 5e-4
MOM = 0.9
conv_channels = []
#inner_lin_layers = [128, 256, 512, 512, 256, 128]
inner_lin_layers = [256, 512, 1024, 1024, 512, 256]
#inner_lin_layers = [256, 512, 1024, 1024, 512, 256]

#res = 227

# generate data
if create_new_data:
    generate_data(lims,
        train_data_size, validation_data_size,
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
input_size = 4
output_size = 4

lin_layers = inner_lin_layers
lin_layers.append(output_size)
lin_layers.insert(0, input_size)

# create network instance
net = NeuralNetwork(conv_channels, lin_layers).to(device)

# create optimizer
#optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOM)
optimizer = optim.Adam(net.parameters(), lr=LR)
#optimizer = optim.Adamax(net.parameters(), lr=LR)
#scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)

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
