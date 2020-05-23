import numpy as np
import gzip
def ld():
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
        #image = [[itm] for itm in image] #this is stupid and bad and i should modify other functions instead of doing this
        labelbuf = labs.read(1)
        label = np.frombuffer(labelbuf, dtype=np.uint8)
        imglst.append((image, label))
        i+=1
    data = np.asarray(imglst)
    del imglst #die
    return data