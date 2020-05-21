import numpy as np
import Functions.LJNNbasic as ljnnb
import Algorithms.Train as train
np.random.seed(0)
sizes = [1,2,3]

weights, biases, delta = ljnnb.init(sizes)

a, ps = ljnnb.feedforwards(weights, biases, [[1]])

do = [[1],[1],[1]]




delta = ljnnb.geterror(a, do, ps, delta, weights)