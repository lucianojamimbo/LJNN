import numpy as np
import Functions.LJNNbasic as ljnnb
import Algorithms.Train as train
import loadmnist as ld
np.random.seed(234)
sizes = [784,256,784] #set the shape of the network
weights, biases, delta = ljnnb.init(sizes) #initialise the weights, biases, and delta randomly but with the correct shape
print("loading train data")
data = ld.ldtrain()
print("train data loaded")
print("loading test data")
testdata = ld.ldtest()
print("test data loaded")
a, ps = ljnnb.feedforwards(weights, biases, data[0][0])
weights, biases = train.SimpleGradientDescent(a, delta, weights, biases, 0.1, 5, data, testdata)


print("testlabel", data[0][1][0])
a, ps = ljnnb.feedforwards(weights, biases, data[0][0])
test = np.reshape(a[-1], (28,28))
import matplotlib.pyplot as plt
plt.imshow(test)