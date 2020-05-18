#collection of basic functions for neural networks.
import numpy as np
def sigmoid(z):
    return 1.0/(1.0+np.exp(np.negative(z)))
def init(sizes):
    biases = [np.random.randn(y, 1) for y in sizes[1:]]
    weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    delta = np.zeros(len(sizes)-1)#one zero for each non-input layer
    return np.array(weights), np.array(biases), np.array(delta)
def feedforwards(w, b, inp):
    a = inp
    activations = []
    presig = []
    activations.append(inp)
    for iw, ib in zip(w, b):
        bs = np.add(np.dot(iw, a), ib)
        print(bs)
        a = sigmoid(bs)
        activations.append(a)
        presig.append(bs)
    return activations, bs