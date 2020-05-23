import numpy as np
import copy 
def SimpleGradientDescent(activations, delta, weights, biases, eta):
    biases = np.subtract(biases, np.multiply(eta, delta))
    nabla_w = copy.deepcopy(activations[1:])
    for i in range(0, len(activations)-1):
        nabla_w[i] = np.multiply(activations[i], delta[i])
    for i in range(0, len(nabla_w)):
        nabla_w[i] = np.reshape(nabla_w[i], np.shape(weights[i]))    
    weights = np.subtract(weights, nabla_w)
    return weights, biases