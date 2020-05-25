import Functions.LJNNbasic as ljnnb
import loadmnist as ld
import pickle
import matplotlib.pyplot as plt
import numpy as np
data = ld.ldtrain()
testdata = ld.ldtest()
encoderweights = pickle.load(open("data/encoderweights.pickle", "rb"))
encoderbiases = pickle.load(open("data/encoderbiases.pickle", "rb"))
decoderweights = pickle.load(open("data/decoderweights.pickle", "rb"))
decoderbiases = pickle.load(open("data/decoderbiases.pickle", "rb"))
def encode(inputdata):
    a, ps = ljnnb.feedforwards(encoderweights, encoderbiases, inputdata)
    return a[-1]
def decode(inputdata):
    a, ps = ljnnb.feedforwards(decoderweights, decoderbiases, inputdata)
    return a[-1]
def compressdataset(data):
    i = 0
    while i < len(data):
        print(i)
        data[i][0] = encode(data[i][0])
        i+=1
    return data
data = compressdataset(data)
with open("Data/compressedMNIST.pickle", "wb") as f:
    pickle.dump(data, f)