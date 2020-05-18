import numpy as np
import Functions.LJNNbasic as ljnnb
np.random.seed(0)
sizes = [1,2,3]
weights, biases, delta = ljnnb.init(sizes)

weights = np.array([np.array([[1],[2]]),np.array([[3,4],[5,6],[7,8]])], dtype=object)
activations, presig = ljnnb.feedforwards(weights, biases, [1])