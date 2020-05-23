#collection of basic functions for neural networks.
import numpy as np
def sigmoid(z):
    return 1/(np.add(1,np.exp(np.negative(z))))
def sigmoidderivative(x):#only ever pass presig into this!
    return (sigmoid(x)*(1-sigmoid(x)))
def init(sizes):
    weights = []
    for i in range(1, len(sizes)):
        weights.append(np.random.randn(sizes[i], sizes[i-1]))
    biases = []
    for i in range(1, len(sizes)):
        biases.append(np.random.randn(sizes[i], 1))
    delta = []
    for i in range(1, len(sizes)):
        delta.append(np.random.randn(sizes[i], 1))
    return weights, biases, delta
def feedforwards(w, b, inp):
    a = inp
    activations = []
    presig = []
    activations.append(inp)
    for iw, ib in zip(w, b):
        x = np.dot(iw, a)
        bs = np.add(x, ib.T[0])
        a = sigmoid(bs)       
        activations.append(a)
        presig.append(bs)
    return activations, presig
def geterror(activations, desiredoutput, presig, delta, weights, biases):#calculate delta
    x = np.subtract(activations[-1], desiredoutput)
    delta[-1] = np.multiply(x, sigmoidderivative(presig[-1]))
    delta[-1] = np.reshape(delta[-1], np.shape(biases[-1]))
    i = 1 #start at one because error in output is already done
    while i < len(delta):
        delta[-i-1] = np.multiply(np.matmul(weights[-i].T, delta[-i]), np.reshape(sigmoidderivative(presig[-i-1]), np.shape(biases[-i-1])))
        i+=1
    return delta
def cost(do, a):
    return np.sum(np.power(np.subtract(do, a), 2))