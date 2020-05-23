import numpy as np
import Functions.LJNNbasic as ljnnb
import Algorithms.Train as train
import loadmnist as ld
np.random.seed(0)
sizes = [784,16,10]
weights, biases, delta = ljnnb.init(sizes)
data = ld.ld()
a, ps = ljnnb.feedforwards(weights, biases, data[0][0])
graph = []
i = 0
while i < 60000:
    do = [0,0,0,0,0,0,0,0,0,0]
    do[data[i][1][0]] = 1
    a, ps = ljnnb.feedforwards(weights, biases, data[i][0])
    delta = ljnnb.geterror(a, do, ps, delta, weights, biases)
    weights, biases, x = train.SimpleGradientDescent(a, delta, weights, biases, 1)
    graph.append(ljnnb.cost(do, a[-1]))
    i+=1    
import matplotlib.pyplot as plt
plt.plot(graph)
plt.show()