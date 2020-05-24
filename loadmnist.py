import numpy as np
import gzip
def ldtrain():
    imgs = gzip.open('Datasets/train-images-idx3-ubyte.gz', 'r')
    labs = gzip.open('Datasets/train-labels-idx1-ubyte.gz', 'r')
    imgs.read(16) #skip some data (this data can be useful but its more work to read it than to just hard code a couple numbers)
    labs.read(8) #skip some data (this data can be useful but its more work to read it than to just hard code a couple numbers)
    imglst = []
    i = 0
    while i < 60000:
        buf = imgs.read(28 * 28 * 1)
        image = np.frombuffer(buf, dtype=np.uint8)
        image = np.divide(image, 255)
        labelbuf = labs.read(1)
        label = np.frombuffer(labelbuf, dtype=np.uint8)
        imglst.append((image, label))
        i+=1
    data = np.asarray(imglst)
    del imglst #die
    return data
def ldtest():
    testimgs = gzip.open('Datasets/t10k-images-idx3-ubyte.gz', 'r')
    testlabs = gzip.open('Datasets/t10k-labels-idx1-ubyte.gz', 'r')
    testimgs.read(16) #skip some data (this data can be useful but its more work to read it than to just hard code a couple numbers)
    testlabs.read(8) #skip some data (this data can be useful but its more work to read it than to just hard code a couple numbers)
    imglst = []
    i = 0
    while i < 10000:
        testbuf = testimgs.read(28 * 28 * 1)
        testimage = np.frombuffer(testbuf, dtype=np.uint8)
        testimage = np.divide(testimage, 255)
        testlabelbuf = testlabs.read(1)
        testlabel = np.frombuffer(testlabelbuf, dtype=np.uint8)
        imglst.append((testimage, testlabel))
        i+=1
    data = np.asarray(imglst)
    del imglst #die
    return data