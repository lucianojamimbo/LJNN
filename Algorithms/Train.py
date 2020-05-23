import numpy as np
import copy 
def SimpleGradientDescent(activations, delta, weights, biases, eta):
    biases = np.subtract(biases, np.multiply(eta, delta))
    x = copy.deepcopy(activations[1:])
    for i in range(1, len(activations)-1):
        x[i] = np.multiply(activations[i], delta[i])
    for i in range(0, len(x)-1):
        x[i] = np.reshape(x[i], np.shape(weights[i]))
    weights = np.subtract(weights, x)
    return weights, biases