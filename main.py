import numpy as np
import Functions.LJNNbasic as ljnnb
import Algorithms.Train as train
np.random.seed(0)
sizes = [1,2,2]
weights, biases, delta = ljnnb.init(sizes)

activations, presig = ljnnb.feedforwards(weights, biases, [1])
delta = ljnnb.geterror(activations, [0,0], presig, delta, weights)
eta = 1

print("before", biases)
train.SimpleGradientDescent(activations, delta, weights, biases, 1)
print("after", biases)