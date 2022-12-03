import numpy as np
from scipy.special import expit, jv
import matplotlib.pyplot as plt

def sigmoid(x):
    return expit(x)

class Network(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)

        rng = np.random.default_rng()
        self.biases = [rng.standard_normal((y_size, 1))
                       for y_size in sizes[1:]]
        self.weights = [rng.standard_normal((y_size, x_size))
                        for x_size, y_size in zip(sizes[:-1], sizes[1:])]

    def g_act(self, x, last=False):
        if last:
            return x
        else:
            return sigmoid(x)

    def g_prime(self, x, last=False):
        if last:
            return 1
        else:
            return sigmoid(x)*(1-sigmoid(x))

    def cost_derivative(self, predicted, y):
        return (predicted - y)

    def feedforward(self, a):
        counter = 1
        end = False
        for b, w in zip(self.biases, self.weights):
            if (counter == self.num_layers-1):
                end = True
                #print("True")

            a = self.g_act(w @ a + b, end)
            counter += 1
        return a

    def evaluate(self, test_data):

        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(y_p==y) for (y_p, y) in test_results)

    def backprop(self, x, y):
        """ Return a tuple "(nabla_b, nabla_w)" representing the
        gradient for the cost function C_x. "nabla_b" and
        "nabla_w" are layer-by-layer lists of numpy arrays, similar
        to "self.biases" and "self.weights"."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x]
        zs = []

        counter = 1
        last = False
        for b, w in zip(self.biases, self.weights):
            if (counter == self.num_layers-1):
                last = True
                #print("True")

            z = w @ activation + b
            zs.append(z)
            activation = self.g_act(z, last)
            activations.append(activation)
            counter += 1

        # backward pass to get delta_L and delta_l
        delta = self.cost_derivative(activations[-1], y) * \
            self.g_prime(zs[-1], last=True)
        nabla_b[-1] = delta
        nabla_w[-1] = delta @ activations[-2].transpose()

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.g_prime(z, last=False)
            delta = self.weights[-l+1].transpose() @ delta * sp
            nabla_b[-l] = delta
            nabla_w[-l] = delta @ activations[-l-1].transpose()
        return (nabla_b, nabla_w)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def SGD(self, train_data, epochs, mini_batch_size, eta, test_data=None):
        n_test = 0
        if test_data: n_test = len(test_data)
        n = len(train_data)
        rng = np.random.default_rng()

        for j in np.arange(epochs):
            rng.shuffle(train_data)
            mini_batches = [
                train_data[k:k+mini_batch_size]
                for k in np.arange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

def test_fun(x):
    return jv(x[0], x[1])

# run stuff

epochs = 100
mini_batch_size = 5
eta = 0.99

rng = np.random.default_rng()

train_data_size = 10000

input_size = 2
output_size = 1

x = np.zeros((input_size, train_data_size))
x[0,:] = rng.integers(0, 3, train_data_size)
x[1,:] = rng.uniform(0, 10, train_data_size)
y = test_fun(x)

training_data_in = [np.reshape(x[:,j], (input_size, 1))
                    for j in np.arange(train_data_size)]
training_data_out = [np.reshape(y[j], (output_size, 1))
                     for j in np.arange(train_data_size)]
training_data = list(zip(training_data_in, training_data_out))

inner_layers_shape = [31,17,13]
layers_shape = inner_layers_shape
layers_shape.insert(0, input_size)
layers_shape.append(output_size)

my_net = Network(layers_shape)
my_net.SGD(training_data, epochs, mini_batch_size, eta)

X1 = np.zeros((input_size, 100))
X1[0,:] = np.arange(0,1,100)
X1[1,:] = np.linspace(0, 10, 100)
P1 = np.zeros((output_size, 100))
X2 = np.zeros((input_size, 100))
X2[0,:] = np.arange(1,2,100)
X2[1,:] = np.linspace(0, 10, 100)
P2 = np.zeros((output_size, 100))
X3 = np.zeros((input_size, 100))
X3[0,:] = np.arange(2,3,100)
X3[1,:] = np.linspace(0, 10, 100)
P3 = np.zeros((output_size, 100))

for i in np.arange(np.size(X1,1)):
    P1[:,i] = my_net.feedforward(np.reshape(X1[:,i], (input_size, 1)))
    P2[:,i] = my_net.feedforward(np.reshape(X2[:,i], (input_size, 1)))
    P3[:,i] = my_net.feedforward(np.reshape(X3[:,i], (input_size, 1)))

plt.plot(X1[1,:], test_fun(X1), 'b')
plt.plot(X1[1,:], P1[0,:], '.r')
plt.figure()
plt.plot(X2[1,:], test_fun(X2), 'b')
plt.plot(X2[1,:], P2[0,:], '.r')
plt.figure()
plt.plot(X3[1,:], test_fun(X3), 'b')
plt.plot(X3[1,:], P3[0,:], '.r')

plt.show()
