import numpy as np
import Functions.LJNNbasic as ljnnb
import Algorithms.Train as train
import loadmnist as ld
import pickle
#load training data
print("loading train data")
data = ld.ldtrain()
print("train data loaded")
#set up and train the network
np.random.seed(234)
sizes = [784,64,10,64,784] #set the shape of the network
weights, biases, delta = ljnnb.init(sizes) #initialise the weights, biases, and delta randomly but with the correct shape
a, ps = ljnnb.feedforwards(weights, biases, data[0][0])
weights, biases = train.EncoderStochasticGradientDescent(a, ps, delta, weights, biases, 1, 10, 50, data, None)
#run a test after training
print("testlabel", data[0][1][0])
a, ps = ljnnb.feedforwards(weights, biases, data[0][0])
test = np.reshape(a[-1], (28,28))
import matplotlib.pyplot as plt
plt.imshow(test)
#convert the one network variables into two variables so there can be an encoder network and a decoder network
encsizes = [784,64,10]
encoderweights, encoderbiases, encoderdelta = ljnnb.init(encsizes)
encoderweights = weights[:2]
encoderbiases = biases[:2]
decsizers = [10,64,784]
decoderweights, decoderbiases, decoderdelta = ljnnb.init(encsizes)
decoderweights = weights[-2:]
decoderbiases = biases[-2:]
#define functions for encoding, decoding, and saving data
def encode(inputdata):
    a, ps = ljnnb.feedforwards(encoderweights, encoderbiases, inputdata)
    return a[-1]
def decode(inputdata):
    a, ps = ljnnb.feedforwards(decoderweights, decoderbiases, inputdata)
    return a[-1]
def exportencoderweights(w):
    with open("Data/encoderweights.pickle", "wb") as f:
        pickle.dump(w, f)
def exportencoderbiases(b):
    with open("Data/encoderbiases.pickle", "wb") as f:
        pickle.dump(b, f)

def exportdecoderweights(w):
    with open("Data/decoderweights.pickle", "wb") as f:
        pickle.dump(w, f)
def exportdecoderbiases(b):
    with open("Data/decoderbiases.pickle", "wb") as f:
        pickle.dump(b, f)
#save the weights and biases we created
exportencoderweights(encoderweights)
exportencoderbiases(encoderbiases)
exportdecoderweights(decoderweights)
exportdecoderbiases(decoderbiases)