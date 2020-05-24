import numpy as np
import Functions.LJNNbasic as ljnnb
import copy
import time
def SimpleGradientDescent(a, delta, weights, biases, eta, epoch_amt, data, testdata):
    print("training with simple gradient descent")
    nwshape = copy.deepcopy(a[1:])
    graph = []
    start = time.time()
    epochs = 0
    while epochs < epoch_amt:
        iters = 0
        while iters < 60000:
            do = np.zeros(10) #set desired output to all zeros
            do[data[iters][1][0]] = 1 #set desired output based on training data label
            a, ps = ljnnb.feedforwards(weights, biases, data[iters][0]) #feedforwards
            delta = ljnnb.geterror(a, do, ps, delta, weights, biases) #feedbackwards
            biases = np.subtract(biases, np.multiply(eta, delta)) #get nabla_b then set biases accordingly
            nabla_w = nwshape #create nabla_w
            for i in range(0, len(a)-1): #find the values for nabla_w
                nabla_w[i] = np.multiply(a[i], delta[i])
            for i in range(0, len(nabla_w)): #reshape nabla_w
                nabla_w[i] = np.reshape(nabla_w[i], np.shape(weights[i]))    
            weights = np.subtract(weights, np.multiply(nabla_w, eta)) #change weights according to nabla_w
            iters+=1 #and repeat
        print("epoch {0} complete".format(epochs))
        graph.append(ljnnb.test(testdata, weights, biases))
        epochs+=1
    end = time.time()
    print("training time:", end-start)
    #creating this graph is helpful
    import matplotlib.pyplot as plt
    plt.plot(graph)
    plt.show()
    return weights, biases #return the new weights and biases
def StochasticGradientDescent(a, ps, delta, weights, biases, eta, batch_size, epoch_amt, data, testdata):
    print("training with stochastic gradient descent")
    nwshape = copy.deepcopy(a[1:])
    start = time.time()
    graph = []
    epochs = 0
    nabla_b_zero = np.zeros(np.shape(ljnnb.geterror(a, np.zeros(10), ps, delta, weights, biases)))
    nabla_w_zero = np.zeros(np.shape(nwshape))
    while epochs < epoch_amt:
        iters = 0
        dataiter = 0
        while iters < 60000/batch_size:
            batchiter = 0
            nabla_b = nabla_b_zero
            nabla_w = nabla_w_zero
            while batchiter < batch_size:
                do = np.zeros(10)
                do[data[dataiter][1]] = 1
                a, ps = ljnnb.feedforwards(weights, biases, data[dataiter][0])
                delta = ljnnb.geterror(a, do, ps, delta, weights, biases)
                current_nabla_b = delta
                current_nabla_w = nwshape #make it a correct shape
                for i in range(0, len(a)-1): #find the vales
                    current_nabla_w[i] = np.multiply(a[i], delta[i])
                for i in range(0, len(current_nabla_w)): #reshape
                    current_nabla_w[i] = np.reshape(current_nabla_w[i], np.shape(weights[i]))
                nabla_w = np.add(nabla_w, np.divide(current_nabla_w, batch_size))
                nabla_b = np.add(nabla_b, np.divide(current_nabla_b, batch_size))
                batchiter +=1
                dataiter +=1
            weights = np.subtract(weights, np.multiply(nabla_w, eta))
            biases = np.subtract(biases, np.multiply(eta, nabla_b))
            iters +=1
        print("epoch {0} complete".format(epochs))
        graph.append(ljnnb.test(testdata, weights, biases))
        epochs +=1
    end = time.time()
    print("training time:", end-start)
    import matplotlib.pyplot as plt
    plt.plot(graph)
    plt.show()
    return weights, biases