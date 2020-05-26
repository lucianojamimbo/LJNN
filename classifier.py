import Functions.LJNNbasic as ljnnb
import loadmnist as ld
import pickle
import time
print("loading weights and biases")
weights = pickle.load(open("data/classifierweights.pickle", "rb"))
biases = pickle.load(open("data/classifierbiases.pickle", "rb"))
print("done")
print("loading data")
data = ld.ldtrain()
testdata = ld.ldtest()
print("done")
start = time.time()
ljnnb.test(testdata, weights, biases)
end = time.time()
print("time taken testing 10000 images:", end-start)