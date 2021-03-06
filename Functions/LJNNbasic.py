#collection of basic functions for feedforwards neural networks.
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
def cost(do, a):
    return np.sum(np.power(np.subtract(do, a), 2))
def test(testdata, weights, biases):
    print("testing")
    correct = 0
    i = 0
    while i < 10000:
        a, ps = feedforwards(weights, biases, testdata[i][0])
        if testdata[i][1][0] == np.argmax(a[-1]):
            correct+=1
        i+=1
    print("test complete")
    print("correctly classified {0} images out of 10000".format(correct))
    return correct