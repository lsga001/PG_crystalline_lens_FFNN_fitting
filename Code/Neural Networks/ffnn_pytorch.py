import os
import numpy as np
import torch
import zernike
import torch.optim as optim
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
#from torchvision import datasets, transforms
from scipy.special import jv
import matplotlib.pyplot as plt

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
                nn.ReLU()
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

epochs = 1000
train_data_size = 1000
validation_data_size = 2000
inner_layer = [31, 17, 13]



x = torch.zeros((train_data_size+validation_data_size, 2), device=device)
y = torch.zeros((train_data_size+validation_data_size, 1), device=device)
x[:,0] = torch.randint(low=0, high=5, size=(train_data_size+validation_data_size,))
x[:,1].uniform_(0,10)
y = jv(x[:,0], x[:,1]).reshape(-1,1)

input_size = x.size(dim=1)
output_size = y.size(dim=1)
output_size = 1

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
optimizer = optim.SGD(net.parameters(), lr=0.05) # ReLU
#optimizer = optim.SGD(net.parameters(), lr=0.5) # Sigmoid

n = train_data_size
bs = int(train_data_size/100) # batch size

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

X0 = torch.zeros((100, input_size), device=device)
Y0 = torch.zeros((100, output_size), device=device)
X1 = torch.zeros((100, input_size), device=device)
Y1 = torch.zeros((100, output_size), device=device)
X2 = torch.zeros((100, input_size), device=device)
Y2 = torch.zeros((100, output_size), device=device)
X3 = torch.zeros((100, input_size), device=device)
Y3 = torch.zeros((100, output_size), device=device)
X4 = torch.zeros((100, input_size), device=device)
Y4 = torch.zeros((100, output_size), device=device)

domain = torch.linspace(0, 10, 100)
X = []
Y = []


X0[:,0] = torch.arange(0, 1, 100)
X0[:,1] = domain
Y0 = jv(X0[:,0], X0[:,1]).reshape(-1, 1)
X1[:,0] = torch.arange(1, 2, 100)
X1[:,1] = domain
Y1 = jv(X1[:,0], X1[:,1]).reshape(-1, 1)
X2[:,0] = torch.arange(2, 3, 100)
X2[:,1] = domain
Y2 = jv(X2[:,0], X2[:,1]).reshape(-1, 1)
X3[:,0] = torch.arange(3, 4, 100)
X3[:,1] = domain
Y3 = jv(X3[:,0], X3[:,1]).reshape(-1, 1)
X4[:,0] = torch.arange(4, 5, 100)
X4[:,1] = domain
Y4 = jv(X4[:,0], X4[:,1]).reshape(-1, 1)

with torch.no_grad():
    P0 = net(X0)
    P1 = net(X1)
    P2 = net(X2)
    P3 = net(X3)
    P4 = net(X4)

plt.figure()
plt.plot(X0[:,1], Y0, 'b')
plt.plot(X0[:,1], P0, '.r')

plt.figure()
plt.plot(X1[:,1], Y1, 'b')
plt.plot(X1[:,1], P1, '.r')

plt.figure()
plt.plot(X2[:,1], Y2, 'b')
plt.plot(X2[:,1], P2, '.r')

plt.figure()
plt.plot(X3[:,1], Y3, 'b')
plt.plot(X3[:,1], P3, '.r')

plt.figure()
plt.plot(X4[:,1], Y4, 'b')
plt.plot(X4[:,1], P4, '.r')

plt.show()
