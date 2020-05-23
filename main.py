import numpy as np
import Functions.LJNNbasic as ljnnb
import Algorithms.Train as train
np.random.seed(0)


#this all works
sizes = [1,2,3]
weights, biases, delta = ljnnb.init(sizes)
a, ps = ljnnb.feedforwards(weights, biases, [0])

graph = []
i = 0
while i < 1000:
    do = [1,1,1]
    delta = ljnnb.geterror(a, do, ps, delta, weights, biases)
    weights, biases = train.SimpleGradientDescent(a, delta, weights, biases, 1)
    a, ps = ljnnb.feedforwards(weights, biases, [0])
    graph.append(a[-1])
    i+=1
    
import matplotlib.pyplot as plt

plt.plot(graph)
plt.show()