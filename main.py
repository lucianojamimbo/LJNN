import numpy as np
import Functions.LJNNbasic as ljnnb
import Algorithms.Train as train
import loadmnist as ld
np.random.seed(234)
sizes = [784,64,10,64,784] #set the shape of the network
weights, biases, delta = ljnnb.init(sizes) #initialise the weights, biases, and delta randomly but with the correct shape
print("loading train data")
data = ld.ldtrain()
print("train data loaded")
print("loading test data")
testdata = ld.ldtest()
print("test data loaded")
a, ps = ljnnb.feedforwards(weights, biases, data[0][0])
weights, biases = train.SimpleGradientDescent(a, delta, weights, biases, 0.1, 15, data, testdata)
print("testlabel", data[0][1][0])
a, ps = ljnnb.feedforwards(weights, biases, data[0][0])
test = np.reshape(a[-1], (28,28))
import matplotlib.pyplot as plt
plt.imshow(test)
encsizes = [784,64,10]
encoderweights, encoderbiases, encoderdelta = ljnnb.init(encsizes)
encoderweights = weights[:2]
encoderbiases = biases[:2]
decsizers = [10,64,784]
decoderweights, decoderbiases, decoderdelta = ljnnb.init(encsizes)
decoderweights = weights[-2:]
decoderbiases = biases[-2:]
def encode(inputdata):
    a, ps = ljnnb.feedforwards(encoderweights, encoderbiases, inputdata)
    return a[-1]
def decode(inputdata):
    a, ps = ljnnb.feedforwards(decoderweights, decoderbiases, inputdata)
    return a[-1]