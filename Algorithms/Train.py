import numpy as np
import Functions.LJNNbasic as ljnnb
import copy
import time
def EncoderSimpleGradientDescent(a, delta, weights, biases, eta, epoch_amt, data, testdata):
    print("training with simple gradient descent")
    nwshape = copy.deepcopy(a[1:])
    graph = []
    start = time.time()
    epochs = 0
    while epochs < epoch_amt:
        np.random.shuffle(data)
        iters = 0
        while iters < 60000:
            do = data[iters][0]
            a, ps = ljnnb.feedforwards(weights, biases, data[iters][0]) #feedforwards
            delta = geterror(a, do, ps, delta, weights, biases) #feedbackwards
            biases = np.subtract(biases, np.multiply(eta, delta)) #get nabla_b then set biases accordingly
            nabla_w = nwshape #create nabla_w
            for i in range(0, len(a)-1): #find the values for nabla_w
                nabla_w[i] = np.multiply(a[i], delta[i])
            for i in range(0, len(nabla_w)): #reshape nabla_w
                nabla_w[i] = np.reshape(nabla_w[i], np.shape(weights[i]))    
            weights = np.subtract(weights, np.multiply(nabla_w, eta)) #change weights according to nabla_w
            costvar = ljnnb.cost(do, a[-1])
            graph.append(costvar)
            iters+=1 #and repeat
        print("epoch {0} complete".format(epochs))
        epochs+=1
    end = time.time()
    print("training time:", end-start)
    import matplotlib.pyplot as plt
    plt.plot(graph)
    plt.show()
    return weights, biases #return the new weights and biases

def EncoderStochasticGradientDescent(a, ps, delta, weights, biases, eta, batch_size, epoch_amt, data, testdata):
    print("training with stochastic gradient descent")
    nwshape = copy.deepcopy(a[1:])
    start = time.time()
    graph = []
    epochs = 0
    nabla_b_zero = np.zeros(np.shape(geterror(a, np.zeros(784), ps, delta, weights, biases)))
    nabla_w_zero = np.zeros(np.shape(nwshape))
    while epochs < epoch_amt:
        np.random.shuffle(data)
        iters = 0
        dataiter = 0
        while iters < 60000/batch_size:
            batchiter = 0
            nabla_b = nabla_b_zero
            nabla_w = nabla_w_zero
            while batchiter < batch_size:
                do = data[dataiter][0]
                a, ps = ljnnb.feedforwards(weights, biases, data[dataiter][0])
                delta = geterror(a, do, ps, delta, weights, biases)
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
            costvar = ljnnb.cost(do, a[-1])
            graph.append(costvar)
            weights = np.subtract(weights, np.multiply(np.divide(eta, batch_size), nabla_w))
            biases = np.subtract(biases, np.multiply(np.divide(eta, batch_size), nabla_b))
            iters +=1
        print("epoch {0} complete".format(epochs))
        epochs +=1
    end = time.time()
    print("training time:", end-start)
    import matplotlib.pyplot as plt
    plt.plot(graph)
    plt.show()
    return weights, biases

def SimpleGradientDescent(a, delta, weights, biases, eta, epoch_amt, data, testdata):
    print("training with simple gradient descent")
    nwshape = copy.deepcopy(a[1:])
    graph = []
    start = time.time()
    epochs = 0
    while epochs < epoch_amt:
        np.random.shuffle(data)
        iters = 0
        while iters < 60000:
            do = np.zeros(10) #set desired output to all zeros
            do[data[iters][1][0]] = 1 #set desired output based on training data label
            a, ps = ljnnb.feedforwards(weights, biases, data[iters][0]) #feedforwards
            delta = geterror(a, do, ps, delta, weights, biases) #feedbackwards
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
    nabla_b_zero = np.zeros(np.shape(geterror(a, np.zeros(10), ps, delta, weights, biases)))
    nabla_w_zero = np.zeros(np.shape(nwshape))
    while epochs < epoch_amt:
        np.random.shuffle(data)
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
                delta = geterror(a, do, ps, delta, weights, biases)
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
            weights = np.subtract(weights, np.multiply(np.divide(eta, batch_size), nabla_w))
            biases = np.subtract(biases, np.multiply(np.divide(eta, batch_size), nabla_b))
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

def geterror(activations, desiredoutput, presig, delta, weights, biases):#calculate delta
    x = np.subtract(activations[-1], desiredoutput)
    delta[-1] = np.multiply(x, ljnnb.sigmoidderivative(presig[-1]))
    delta[-1] = np.reshape(delta[-1], np.shape(biases[-1]))
    i = 1 #start at one because error in output is already done
    while i < len(delta):
        delta[-i-1] = np.multiply(np.matmul(weights[-i].T, delta[-i]), np.reshape(ljnnb.sigmoidderivative(presig[-i-1]), np.shape(biases[-i-1])))
        i+=1
    return delta