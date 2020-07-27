from influence.influence.smooth_hinge import SmoothHinge
from influence.influence.dataset import DataSet
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base

# Load libraries
import pandas
import numpy as np
#import autograd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import defenses
import iterative_attack
import random


import os

def load_dataset():
    # Load dataset
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


def computeEuclidianDist(X, Y , centroid):
        print(X.shape[0])
        distance = np.zeros(X.shape[0])
        #print(distance)
        #print(centroid.values)
        names = list(centroid.index)
        #print(list(centroid.index))
        print(Y)
        for y in set(Y):
                tc = centroid.values[names.index(y),:]
                tc = tc.reshape(1,-1)
                distance[Y == y] = metrics.pairwise.pairwise_distances(X[Y == y, :], \
                                                                        tc, metric = 'euclidean').ravel()
        print(distance)
        return distance

def filterData(X, Y, distance, percentageToRemove):
        #receive dataset X and Y, sort by the distance and remove the % farthest away
        percentageToKeep = 1 - percentageToRemove

        idx_to_keep = []
        #print(type(idx_to_keep))
        for y in set(Y): 
                num_to_keep = int(np.round(percentageToKeep * np.sum(Y == y)))

                idx_to_keep.append( \
                    np.where(Y == y)[0][np.argsort(distance[Y == y])[:num_to_keep]])

        idx_to_keep = np.concatenate(idx_to_keep)

        #modify to keep the ordering consistent
        mask = np.zeros(X.shape[0])
        for i in idx_to_keep:
            mask[i] = 1

        #print(mask)
        #print("mask result:")
        #print(X[mask == 1])
        X_def = X[mask == 1]
        Y_def = Y[mask == 1]
        #print(X_def)
        #print(Y_def)
        #print(idx_to_keep)
        return X_def, Y_def, idx_to_keep


dataset = pandas.DataFrame(np.array([[0,0,0],[1,1,0],[2,3,1],[4,4,1]]), columns=['a','b','class'])
print(dataset)
median = dataset.groupby('class').median()
print(median)
    
x = np.array([[0,0], [2,3], [1,1], [4,4]])
y = np.array([0,1,0,1])



dist = computeEuclidianDist(x,y,median)
fx,fy,index = filterData(x,y,dist,0)
print(fx)
print(fy)
