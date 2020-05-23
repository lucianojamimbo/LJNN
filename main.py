import numpy as np
import Functions.LJNNbasic as ljnnb
import Algorithms.Train as train
import loadmnist as ld
np.random.seed(0)
def test():
    print("testing")
    correct = 0
    i = 0
    while i < 10000:
        a, ps = ljnnb.feedforwards(weights, biases, data[i][0])
        if data[i][1][0] == np.argmax(a[-1]):
            correct+=1
        i+=1
    print("test complete")
    return correct    
sizes = [784,32,10]
weights, biases, delta = ljnnb.init(sizes)
print("loading train data")
data = ld.ld()
print("train data loaded")
a, ps = ljnnb.feedforwards(weights, biases, data[0][0])
graph = []
epochs = 0
while epochs < 20:
    i = 0
    while i < 60000:
        do = [0,0,0,0,0,0,0,0,0,0]
        do[data[i][1][0]] = 1
        a, ps = ljnnb.feedforwards(weights, biases, data[i][0])
        delta = ljnnb.geterror(a, do, ps, delta, weights, biases)
        weights, biases = train.SimpleGradientDescent(a, delta, weights, biases, 0.5)
        i+=1  
    print("epoch {0} complete".format(epochs))
    graph.append(test())
    epochs+=1
import matplotlib.pyplot as plt
plt.plot(graph)
plt.show()
print("amount of images correctly classified out of 10000:", test())