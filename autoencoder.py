import Functions.LJNNbasic as ljnnb
import loadmnist as ld
import pickle
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