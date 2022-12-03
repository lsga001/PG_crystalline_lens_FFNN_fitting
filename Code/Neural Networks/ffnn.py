import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy import special
import graphviz
from graphviz import Digraph
import decimal

def cost_fun(s, y):
    c = 0.5*np.abs(y-s)**2
    return c

def network_evaluate(X, W, B, g_act, g_out):
    m = len(W) # Number of layers

    O = []
    O.append(X)


    for l in np.arange(0, m-1): # hidden layer number -1
        #print("***")
        #print(np.shape(W[l]))
        #print(np.shape(O[l]))
        #print(np.shape(B[l]))
        #print("***")
        O.append(g_act(W[l]@O[l] + B[l])) # O[l+1]
    O.append(g_out(W[m-1]@O[m-1] + B[m-1]))
    return O

def net_batch_eval(X, W, B, g_act, g_out):
    Y = np.zeros((np.size(X,0),))
    for t in np.arange(0, np.size(X,0)):
        Y[t,] = network_evaluate(X[t,], W, B, g_act, g_out)[-1]
    return Y

def network_update(InOutPairs, W, B, g_act, g_out, alpha, epochs):
    m = len(W)

    for t in np.arange(epochs):
        X, Y = InOutPairs[t]
        error_total = []
        nabla_w = [np.zeros(w.shape) for w in W]
        nabla_b = [np.zeros(b.shape) for b in B]
        for j in np.arange(np.size(Y,-1)):
            O = []
            O = network_evaluate(X[j][np.newaxis].T, W, B, g_act, g_out)
            error_total.append(cost_fun(O[-1], Y[j][np.newaxis].T))

            grad_aL_C = (O[-1] - Y[j][np.newaxis].T) # derivative of error fun

            delta = []
            #delta.append(grad_aL_C * O[-1]*(1-O[-1])) # aplica para g_out sigmoide
            delta.append(grad_aL_C * 1) # usando función activación final sin nada g_out(x) = x

            counter = 0
            for l in np.arange(m-1, -1, -1):
                #print(l)
                #print(np.shape(O[l]))
                #print(np.shape(W[l]))
                #print(np.shape(delta[counter]))
                delta.append( (O[l]*(1-O[l])) * (W[l].T @ delta[counter]))
                #delta.append( (np.diagflat(O[l]*(1.0-O[l])) @ W[l].T) @ delta[counter])
                counter += 1

            delta.reverse()

            #delta_cum = delta*0
            for l in np.arange(0,m):
                nabla_w[l] += delta[l+1] @ O[l].T
                nabla_b[l] += delta[l+1]



            #for l in np.arange(0, m):
                #print(np.shape(W[l]))
                #print(np.shape(delta[l]))
                #print(np.shape(O[l].T))
                #print(np.shape(B[l]))
                #print("")
            #    W[l] -= alpha * (delta[l+1] @ O[l].T)
            #    B[l] -= alpha * delta[l+1] * 1
        for l in np.arange(0,m):
            W[l] -= alpha * nabla_w[l]
            B[l] -= alpha * nabla_b[l]
        print("Error: ", np.sum(error_total))

    return W, B

def network_train():
    return

def g_a(X):
    return expit(X)

def g_o(X):
    return X
    #return g_a(X)

def test_fun(X):
    #print(X[0,])
    #print(X[1,])
    #return special.jv(X[0,],X[1,])
    #return X[1,]**3
    #return X[1,]**1
    return np.cos(10*X[1])


def net_view(W, B):
    f = Digraph('F', filename='net_vis.gv', comment='Neural network viz')
    f.attr(rankdir='LR', size='8,5')
    f.attr('node', shape='doublecircle', style='filled', fillcolor='blue')

    for node in np.arange(1, np.size(W[0],1)+1):
        layer = 0
        f.node('N_{},{}'.format(layer,node), label='')

    f.attr('node', shape='circle', fillcolor='white')
    for layer in np.arange(0,len(B)):
        #with f.subgraph(name='cluster_{}'.format(layer)) as c:
        with f.subgraph(name='cluster_{}'.format(layer)) as c:
            c.attr(color='invis')
            #c.node('N_{},{}'.format(layer+1, 0))
            for node in np.arange(1,np.size(B[layer],0)+1):
                #f.node('N_{},{}'.format(layer+1, node))
                c.node('N_{},{}'.format(layer+1, node), label='')

    for layer in np.arange(0, len(W)):
        for node_prev in np.arange(1, np.size(W[layer],1)+1):
            for node_next in np.arange(1, np.size(W[layer],0)+1):
                f.edge('N_{},{}'.format(layer, node_prev),
                       'N_{},{}'.format(layer+1, node_next))#,
                       # label="{:.2f}".format(W[layer][node_next-1, node_prev-1]))

    f.attr('node', style='filled', fillcolor='red')
    for layer in np.arange(0, len(B)):
        with f.subgraph(name='cluster_{}'.format(layer)) as c:
            #c.attr(color='invis')
            f.node('N_{},{}'.format(layer, 0), label='')
            for node_next in np.arange(1, np.size(W[layer],0)+1):
                f.edge('N_{},{}'.format(layer, 0),
                       'N_{},{}'.format(layer+1, node_next))#,
                #       label="".format(B[layer][node_prev-1]))
    f.view()
    return


from numpy.random import default_rng
rng = default_rng()

# Precision to use
#decimal.getcontext().prec = 100

alpha = 1.0
epochs = 10000
#epochs = 100

x = np.linspace(0, 1, 100)
y = test_fun(x)


# Generation of test cases
InOutPairs = []

x = np.array([rng.integers(0, 1, 100), rng.uniform(0, 1, 100)])
y = test_fun(x)
x = x.T
#InOutPairs.append([x, y])

#print(np.shape(x))
for t in np.arange(epochs):
    x = np.array([rng.integers(0, 1, 100), rng.uniform(0, 1, 100)])
    y = test_fun(x)[np.newaxis].T
    x = x.T
    InOutPairs.append([x, y])



#inner_layers_shape = [31,17,13]
inner_layers_shape = [4,3,5,2]
#inner_layers_shape = [3]


layers_shape = inner_layers_shape
layers_shape.insert(0,np.size(x[0],-1))
#layers_shape.append(np.size(y[0],-1))
layers_shape.append(1)
# layers_shape is the entire shape of layers
# including output layer and input layer
print(layers_shape)

num_layers = len(layers_shape)
W = []
B = []

for i in np.arange(1, num_layers):
    W.append(rng.uniform(-1, -1, (layers_shape[i], layers_shape[i-1])))
    B.append(rng.uniform(-1, -1, (layers_shape[i], 1)))


f_train = test_fun
W, B = network_update(InOutPairs, W, B, g_a, g_o, alpha, epochs)


#O = network_evaluate(np.array([0,5]).T, W, B, g_a, g_o)
#print(O[-1])

net_view(W, B)

x = np.array([np.linspace(0,1,100), np.linspace(0, 1, 100)])
y = test_fun(x)

Y = net_batch_eval(x[np.newaxis].T, W, B, g_a, g_o)

plt.plot(x[1], y, 'b')
#plt.plot(X[:,1], np.array(Sols), '.r')
plt.plot(x[1], Y, '.r')
plt.show()
