import pandas
import numpy as np
import tensorflow as tf

def load_dataset_spambase():
    # Load dataset from url
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    dataset = pandas.read_csv(url, header = None)

    features = []
    for i in range (0,57):
        dataset.rename(columns = {i: 'feature'+str(i)}, inplace=True)
        features.append('feature'+str(i))
    dataset.rename(columns = {57: 'class'}, inplace=True)
    #change class labels to 1 and -1
    dataset['class'].replace(0,-1, inplace = True)
    #print(dataset)

    return dataset

def load_dataset_mnist17(sampling=False):
    #load dataset from tensorflow keras datasets
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

    #X_train = np.vstack([img.reshape(-1,) for img in mnist.train.images])
    #Y_train = mnist.train.labels

    #X_test = np.vstack([img.reshape(-1,) for img in mnist.test.images])
    #Y_test = mnist.test.labels

    X_train = X_train.reshape(60000, 784).astype('float32') 
    X_test = X_test.reshape(10000, 784).astype('float32') 

    ones = (Y_train == 1)
    sevens  = (Y_train == 7)
    combined_mask = np.ma.mask_or(ones, sevens)
    
    X_train = X_train[combined_mask]
    Y_train = Y_train[combined_mask]
    Y_train = np.where(Y_train==7, -1 ,Y_train)

    #sampling to cut down running time
    if sampling == True:
        X_ones= X_train[Y_train==1,:]
        X_zeros = X_train[Y_train==-1,:]
        #returns the sampled points
        samples_one = X_ones[np.random.choice(X_ones.shape[0], int(X_ones.shape[0]/2), replace=False),:]
        samples_zero = X_zeros[np.random.choice(X_zeros.shape[0], int(X_zeros.shape[0]/2), replace=False),:]
        X_train = np.concatenate((samples_one,samples_zero), axis=0)
        Y_train = np.concatenate( (np.ones(int(X_ones.shape[0]/2)),
                                   np.zeros(int(X_zeros.shape[0]/2))
                                   ) )

        #shuffle the array
        X_train, Y_train = shuffle_arrays(X_train, Y_train)
        #map back to -1
        Y_train = np.where(Y_train==0, -1 ,Y_train)

    ones_t = (Y_test == 1)
    sevens_t  = (Y_test == 7)
    combined_mask_t = np.ma.mask_or(ones_t, sevens_t)

    X_test = X_test[combined_mask_t]
    Y_test = Y_test[combined_mask_t]
    Y_test = np.where(Y_test==7, -1 ,Y_test)
    
    #print(X_train.shape)
    #print(Y_train.shape)
    #for i in range (1):
        #print(X_train[i])
    #print(Y_train)

    return X_train, Y_train, X_test, Y_test

def load_dataset_mnist01(sampling = False):
    #load dataset from tensorflow keras datasets
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

    #X_train = np.vstack([img.reshape(-1,) for img in mnist.train.images])
    #Y_train = mnist.train.labels

    #X_test = np.vstack([img.reshape(-1,) for img in mnist.test.images])
    #Y_test = mnist.test.labels

    X_train = X_train.reshape(60000, 784).astype('float32') 
    X_test = X_test.reshape(10000, 784).astype('float32') 

    ones = (Y_train == 1)
    zeros  = (Y_train == 0)
    combined_mask = np.ma.mask_or(ones, zeros)
    
    X_train = X_train[combined_mask]
    Y_train = Y_train[combined_mask]
    Y_train = np.where(Y_train==0, -1 ,Y_train)

    #The Curie setup requires sampling a subset of points randomly
    if sampling == True:
        X_ones= X_train[Y_train==1,:]
        X_zeros = X_train[Y_train==-1,:]
        #returns the sampled points
        samples_one = X_ones[np.random.choice(X_ones.shape[0], 1250, replace=False),:]
        samples_zero = X_zeros[np.random.choice(X_zeros.shape[0], 1250, replace=False),:]
        X_train = np.concatenate((samples_one,samples_zero), axis=0)
        Y_train = np.concatenate( (np.ones(1250),np.zeros(1250)) )

        #shuffle the array
        X_train, Y_train = shuffle_arrays(X_train, Y_train)
        #map back to -1
        Y_train = np.where(Y_train==0, -1 ,Y_train)

    ones_t = (Y_test == 1)
    zeros_t  = (Y_test == 0)
    combined_mask_t = np.ma.mask_or(ones_t, zeros_t)

    X_test = X_test[combined_mask_t]
    Y_test = Y_test[combined_mask_t]
    Y_test = np.where(Y_test==0, -1 ,Y_test)
    
    #print(X_train.shape)
    #print(Y_train.shape)
    #for i in range (1):
        #print(X_train[i])
    #print(Y_train)

    return X_train, Y_train, X_test, Y_test

def shuffle_arrays(X, Y):
    #assert len(a) == len(b)
    p = np.random.permutation(Y.shape[0])
    return X[p], Y[p]
