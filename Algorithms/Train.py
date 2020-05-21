import numpy as np
import copy
def SimpleGradientDescent(activations, delta, weights, biases, eta):
    weights = np.subtract(weights, np.multiply(eta, np.multiply(activations[1:], delta)))
    biases = np.subtract(biases, np.multiply(eta, delta))
    return weights, biases