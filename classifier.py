import numpy as np
import Functions.LJNNbasic as ljnnb
import Algorithms.Train as train
import loadmnist as ld
np.random.seed(234)
sizes = [784,32,10] #set the shape of the network
weights, biases, delta = ljnnb.init(sizes) #initialise the weights, biases, and delta randomly but with the correct shape
print("loading train data")
data = ld.ldtrain()
print("train data loaded")
print("loading test data")
testdata = ld.ldtest()
print("test data loaded")
a, ps = ljnnb.feedforwards(weights, biases, data[0][0])
weights, biases = train.StochasticGradientDescent(a, ps, delta, weights, biases, 4, 10, 30, data, testdata)
print("amount of images correctly classified out of 10000:", ljnnb.test(testdata, weights, biases))