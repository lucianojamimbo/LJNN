#collection of basic functions for neural networks.
import numpy as np
def sigmoid(z):
    return 1/(np.add(1,np.exp(np.negative(z))))
def sigmoidderivative(x):#only ever pass presig into this!
    return (sigmoid(x)*(1-sigmoid(x)))
def init(sizes):
    biases = [np.random.randn(y, 1) for y in sizes[1:]]
    weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    delta = np.zeros(len(sizes)-1)#one zero for each non-input layer
    delta = [[itm] for itm in delta]
    return weights, biases, delta
def feedforwards(w, b, inp):
    a = inp
    activations = []
    presig = []
    activations.append(inp)
    for iw, ib in zip(w, b):
        x = np.dot(iw, a)
        bs = np.add(x, [item for sublist in ib for item in sublist])
        a = sigmoid(bs)
        activations.append(a)
        presig.append(bs)
    return activations, presig
def geterror(activations, desiredoutput, presig, delta, weights):#calculate delta
    delta[-1] = np.multiply((np.subtract(activations[-1], desiredoutput)), sigmoidderivative(presig[-1]))
    i = 1 #start at one because error in output is already done
    while i < len(delta):
        delta[-i-1] = np.multiply(np.transpose(np.matmul(weights[-i].T, delta[-i])), sigmoidderivative(presig[-i-1]))
        
        
        
        
        
        i+=1
    return delta
