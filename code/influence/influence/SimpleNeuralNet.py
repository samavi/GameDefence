from sklearn import preprocessing
import numpy as np
def derivative(x):
    return x * (1.0 - x)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def log_loss(x, t):
    exponents = -(x-1)/t
    # exponents = -(x)/t
    max_elems = tf.maximum(exponents, tf.zeros_like(exponents))

    return t * (max_elems + tf.log(
        tf.exp(exponents - max_elems) + 
        tf.exp(tf.zeros_like(exponents) - max_elems)))
    # return t * tf.log(tf.exp(-(x)/t) + 1)        


class SimpleNeuralNet():
    def __init__(self, X_train, Y_train, X_validation, Y_validation, seed = 8, numLayers = 3, nodesEachLayer = [57,4,1],learning_rate = 0.001 ):
        #data sets
        self.X_train = X_train
        self.Y_train = Y_train.reshape(-1,1)
        self.X_validation = X_validation
        self.Y_validation = Y_validation.reshape(-1,1)
        self.learning_rate = learning_rate

        self.mean = np.mean(self.X_train, axis = 0)
        self.variance = np.var(self.X_train, axis = 0)

        print(self.X_train)
        #map Y back to 0/1
        self.mapY()
        #feature scale
        self.featureScale()
        #print(np.var(self.X_train, axis = 0).shape)
        #print(np.mean(self.X_train, axis = 0).shape)
        #print(self.X_train)
        #print((self.X_train[0] - np.mean(self.X_train, axis = 0) ) / np.var(self.X_train, axis = 0))
        #print(self.Y_validation)
        
        self.numLayers = numLayers

        #randomize neural net weights
        np.random.seed(seed)
        #print(X_train.shape[1])
        assert(nodesEachLayer[0] == X_train.shape[1])
        self.weights = []
        for i in range(numLayers - 1):
            weight = 2 * np.random.random((nodesEachLayer[i], nodesEachLayer[i+1])) - 1
            self.weights.append(weight)
        
            
        
    def mapY(self):
        self.Y_train = np.where(self.Y_train == -1, 0, self.Y_train)
        self.Y_validation = np.where(self.Y_validation == -1, 0, self.Y_validation)
        
    def featureScale(self):
        self.X_train = preprocessing.scale(self.X_train)
        self.X_validation = preprocessing.scale(self.X_validation)
        
    def update_train(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train.reshape(-1,1)
        #reset mean and variance
        self.mean = np.mean(self.X_train, axis = 0)
        self.variance = np.var(self.X_train, axis = 0)
        #map Y back to 0/1
        self.mapY()
        #feature scale
        self.featureScale()
        
    def train(self, iteration = 1000):
        #dim1 = len(X_train[0])
        #dim2 = 4
        # randomly initialize the weight vectors
        #np.random.seed(1)
        #weight0 = 2 * np.random.random((dim1, dim2)) — 1
        #weight1 = 2 * np.random.random((dim2, 1)) — 1
        # you can change the number of iterations
        for j in range(iteration):
            # first evaluate the output for each training email
            #start with the input layer
            layers = []
            layer_0 = self.X_train
            layers.append(layer_0)
            
            for i in range(1, self.numLayers):
                #compute the layer's output and store it
                
                currentLayer = sigmoid(np.dot(layers[-1], self.weights[i-1]))
                layers.append(currentLayer)

            #print(layers[1].shape)
            #print(layers[2].shape)
            #layer_1 = sigmoid(np.dot(layer_0,weight0))
            #layer_2 = sigmoid(np.dot(layer_1,weight1))
            #print(layers[-1].shape)
            #print(self.Y_train.shape)
            # calculate the error 
            error_list = []
            error_last_layer = self.Y_train - layers[-1]
            error_list.append(error_last_layer)
            #print("error")
            #print(max(error_list[0]))
            #print(layers[2])

            #print(error_list[-1].shape)
            # perform back propagation 
            for i in range(self.numLayers - 2, -1, -1):
                current_layer_delta = error_list[-1] * derivative(layers[i+1])
                #print(i)
                #print(current_layer_delta)
                if i > 0:
                    current_layer_error = current_layer_delta.dot(self.weights[i].T ) #used for back propagation
                    error_list.append(current_layer_error)
                self.weights[i] += self.learning_rate * layers[i].T.dot(current_layer_delta)
            #print(self.weights)
            #layer_2_delta = layer_2_error * derivative(layer_2)
            #layer_1_error = layer_2_delta.dot(weight1.T)
            #layer_1_delta = layer_1_error * derivative(layer_1)
            # update the weight vectors
            #weight1 += layer_1.T.dot(layer_2_delta)
            #weight0 += layer_0.T.dot(layer_1_delta)
            if j%999 ==0:
                self.validate(self.X_train, self.Y_train)
                self.validate(self.X_validation, self.Y_validation)

    
    def validate(self, X_validation, Y_validation):
        # evaluation on the testing data
        layers = []
        layer_0 = X_validation
        layers.append(layer_0)
            
        for i in range(1, self.numLayers):
            #compute the layer's output and store it
            currentLayer = sigmoid(np.dot(layers[-1], self.weights[i-1]))
            layers.append(currentLayer)
        correct = 0
        # if the output is > 0, then label as spam else no spam
        for i in range(len(layers[-1])):
            if(layers[-1][i][0] > 0.5):
                layers[-1][i][0] = 1
            else:
                layers[-1][i][0] = 0
            if(layers[-1][i][0] == Y_validation[i][0]):
                correct += 1
        # printing the output
        print("total = " +str(len(layers[-1])))
        print("correct = "+str(correct))
        print("accuracy = "+str(correct * 100.0 / len(layers[-1])))
        return correct * 100.0 / len(layers[-1])

    def get_test_accuracy(self):
        return self.validate(self.X_validation, self.Y_validation)
    def get_train_accuracy(self):
        return self.validate(self.X_train, self.Y_train)
    def get_loss(self, data_set, i, feature_scale = False):
        if feature_scale == True:
            x = (data_set.x[i] - self.mean) / self.variance
        else:
            x = data_set.x[i]
        y = data_set.labels[i]
        layers = []        
        layer_0 = x
        layers.append(layer_0)
        for i in range(1, self.numLayers):
            #compute the layer's output and store it
                
            currentLayer = sigmoid(np.dot(layers[-1], self.weights[i-1]))
            layers.append(currentLayer)

        error_last_layer = np.absolute(y - layers[-1])
        return error_last_layer
