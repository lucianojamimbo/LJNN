import numpy as np
import Functions.LJNNbasic as ljnnb
import Algorithms.Train as train
import loadmnist as ld
import pickle
np.random.seed(0)


print("loading train data")
data = ld.ldtrain()
print("train data loaded")
print("loading test data")
testdata = ld.ldtest()
print("test data loaded")


def fromscratch(data, eta, batch_size, epochamount):
    sizes = [784,32,10] #set the shape of the network
    weights, biases, delta = ljnnb.init(sizes) #initialise the weights, biases, and delta randomly but with the correct shape
    a, ps = ljnnb.feedforwards(weights, biases, data[0][0])
    weights, biases = train.StochasticGradientDescent(a, ps, delta, weights, biases, eta, batch_size, epochamount, data, testdata)
    ljnnb.test(testdata, weights, biases)
    return weights, biases
def fromsaved(data, eta, batch_size, epochamount):
    print("loading weights and biases")
    weights = pickle.load(open("data/classifierweights.pickle", "rb"))
    biases = pickle.load(open("data/classifierbiases.pickle", "rb"))
    print("done")
    
    sizes = [784,32,10] #set the shape of the network
    tb, tw, delta = ljnnb.init(sizes) #initialise the weights, biases, and delta randomly but with the correct shape
    ljnnb.test(testdata, weights, biases)
    a, ps = ljnnb.feedforwards(weights, biases, data[0][0])
    weights, biases = train.StochasticGradientDescent(a, ps, delta, weights, biases, eta, batch_size, epochamount, data, testdata)
    ljnnb.test(testdata, weights, biases)
    return weights, biases

def exportweights(w):
    with open("Data/classifierweights.pickle", "wb") as f:
        pickle.dump(w, f)
def exportbiases(b):
    with open("Data/classifierbiases.pickle", "wb") as f:
        pickle.dump(b, f)


weights, biases = fromscratch(data, 3, 10, 100)

exportweights(weights)
exportbiases(biases)