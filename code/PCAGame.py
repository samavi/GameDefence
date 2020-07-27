#REPLICATES PCA GAME ANALYSIS EXPERIMENT- PURE STRATEGY
from influence.influence.smooth_hinge import SmoothHinge
from influence.influence.SmoothHingeElevated import SmoothHingeElevated
from influence.influence.dataset import DataSet
from influence.influence.SimpleNeuralNet import SimpleNeuralNet
import load_dataset

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
import write_result
import iterative_attack
import random
import math


import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def computePCARadii(X_clean, Y_clean, pcas,scalers, labels, percentage_to_keep, num_malicious):
    distance = computePCADistance(X_clean, Y_clean, pcas,scalers, labels)

        #find the max pca distance value for each class
    radii = []
    
    for y in labels:
        y_tmp = np.sort(distance[Y_clean == y])
        num = max(0, y_tmp.shape[0] - int(np.round((y_tmp.shape[0]+num_malicious) * (1-percentage_to_keep))))
        if num>=y_tmp.shape[0]:
            #default to the largest distance
            radii.append(1 * y_tmp[y_tmp.shape[0] - 1])
            
            #radii.append(1000)
        else:
            radii.append(y_tmp[num])

    print("Radii:")
    print(radii)

    return radii
def get_projection_fn(X_clean, Y_clean, pcas,scalers, labels, percentage_to_keep, center, num_malicious, sphere = True):

    class_map, centroids, centroid_vec, sphere_radii, slab_radii= get_data_params(X_clean, Y_clean, 100 * (percentage_to_keep))

    #find the max pca distance value for each class
    radii = computePCARadii(X_clean, Y_clean, pcas,scalers, labels, percentage_to_keep, num_malicious)
    
    def project_onto_feasible_set(X, Y):
        if sphere:
            X = project_onto_sphere(X, Y, sphere_radii, centroids, class_map)
        else:
            X = project_onto_pca(X,Y, radii, pcas,scalers, labels, center)
        return X
    return project_onto_feasible_set

def project_onto_sphere(X, Y, radii, centroids, class_map):

    for y in set(Y):
        idx = class_map[y]        
        radius = radii[idx]
        centroid = centroids[idx, :]

        shifts_from_center = X[Y == y, :] - centroid
        dists_from_center = np.linalg.norm(shifts_from_center, axis=1)

        shifts_from_center[dists_from_center > radius, :] *= radius / np.reshape(dists_from_center[dists_from_center > radius], (-1, 1))
        X[Y == y, :] = shifts_from_center + centroid

        print("Number of (%s) points projected onto sphere: %s" % (y, np.sum(dists_from_center > radius)))

    return X

def get_class_map():
    return {-1: 0, 1: 1}

def get_centroids(X, Y, class_map):
    num_classes = len(set(Y))
    num_features = X.shape[1]
    centroids = np.zeros((num_classes, num_features))
    for y in set(Y):            
        centroids[class_map[y], :] = np.mean(X[Y == y, :], axis=0)
    return centroids
def get_centroid_vec(centroids):
    assert centroids.shape[0] == 2
    centroid_vec = centroids[0, :] - centroids[1, :]
    centroid_vec /= np.linalg.norm(centroid_vec)
    centroid_vec = np.reshape(centroid_vec, (1, -1))
    return centroid_vec


def project_onto_pca(X,Y, radii, pcas,scalers, labels, center):
    drag = 0.05
    #centroid has all 0 as feature values
    pca_dist = computePCADistance(X, Y, pcas,scalers, labels)
    
    euc_dist = computeEuclidianDist(X, Y , center)
    
    count = 0
    for y in labels:
        if X[ Y == y].shape[0] == 0:
            count+=1
            continue
        
        euc_dist_curr = euc_dist[Y == y]
        radius = radii[count]
        #print("Original point: ")
        #print(X[Y==y])
        
        #scale data and convert into pca space
        scaler = scalers[count]
        #acquire pca projection
        x_proj = pcas[count].transform(scaler.transform(X[ Y == y]))
        #print(x_proj)
        shift_from_center = x_proj
        class_distance = pca_dist[Y==y]
        #print(class_distance)
        #print(class_distance)
        vector_magnitude = np.linalg.norm(shift_from_center, axis = 1)

        #apply drag force on points
        shift_from_center += drag * np.sign(shift_from_center) * pcas[count].explained_variance_
        #compute unit vector of each out-of-bound datapoint
        unit_vecs = shift_from_center[class_distance > radius, :] / vector_magnitude[class_distance > radius, np.newaxis] 
        #print(unit_vecs.shape)
        #finally, project the point onto the edge of the ellipse

        #fancy mathmatical formula
        multipliers = np.sqrt(radius / np.sum(np.divide(np.square(unit_vecs), np.square(pcas[count].explained_variance_)), axis = 1))

        shift_from_center[class_distance > radius, :] = np.multiply(unit_vecs, multipliers[:,np.newaxis]) 
        

        #print("Projected point: ")
        #print(shift_from_center[class_distance > radius, :])
        #print(shift_from_center)
        #print("PCA distance:")
        #print(np.sum(np.divide(np.square(shift_from_center), np.square(pcas[count].explained_variance_)), axis = 1))
        #un-transform the points back to normal coordinate
        shift_from_center= pcas[count].inverse_transform(shift_from_center)
        #un-scale the data
        shift_from_center= scaler.inverse_transform(shift_from_center)

        
        X[Y == y] = shift_from_center
        #print("Inverted point: ")
        #print(X[Y==y])
        count+=1
        print("Number of (%s) points projected onto sphere: %s" % (y, np.sum(class_distance > radius)))
        print("Euclidian distance of projected points:")
        print(euc_dist_curr[class_distance>radius])
    return X

def get_data_params(X, Y, percentile):    
    num_classes = len(set(Y))
    num_features = X.shape[1]
    centroids = np.zeros((num_classes, num_features))
    class_map = get_class_map()
    centroids = get_centroids(X, Y, class_map)

    # Get radii for sphere
    sphere_radii = np.zeros(2)
    dists = defenses.compute_dists_under_Q(    
        X, Y,
        Q=None,
        centroids=centroids,
        class_map=class_map,    
        norm=2)
    for y in set(Y):            
        sphere_radii[class_map[y]] = np.percentile(dists[Y == y], percentile)

    # Get vector between centroids
    centroid_vec = get_centroid_vec(centroids)

    # Get radii for slab
    slab_radii = np.zeros(2)
    for y in set(Y):            
        dists = np.abs( 
            (X[Y == y, :].dot(centroid_vec.T) - centroids[class_map[y], :].dot(centroid_vec.T)))            
        slab_radii[class_map[y]] = np.percentile(dists, percentile)

    return class_map, centroids, centroid_vec, sphere_radii, slab_radii



def copy_random_points(X, Y, mask_to_choose_from=None, target_class=1, num_copies=1, 
                       random_seed=18, replace=False):
    # Only copy from points where mask_to_choose_from == True

    np.random.seed(random_seed)    
    combined_mask = (np.array(Y, dtype=int) == target_class)
    if mask_to_choose_from is not None:
        combined_mask = combined_mask & mask_to_choose_from

    idx_to_copy = np.random.choice(
        np.where(combined_mask)[0],
        size=num_copies,
        replace=replace)

##    if sparse.issparse(X):
##        X_modified = sparse.vstack((X, X[idx_to_copy, :]))
##    else:
    X_modified = np.append(X, X[idx_to_copy, :], axis=0)
    Y_modified = np.append(Y, Y[idx_to_copy])
    return X_modified, Y_modified


def PCA_analysis(dataset):
    #only need to take X as param
    
    # (value - mean) / variance
    scalar = StandardScaler() 
    
    scalar.fit(dataset) 
    scaled_data = scalar.transform(dataset)

    # keep all pca components
    pca = PCA() 
    pca.fit(scaled_data) 
     
  
    #print(x_pca.shape)
    print("Variance: ")
    print(pca.explained_variance_)
    #return the entire PCA analysis
    #use "x_pca = pca.transform(scaled_data)" to transform
    #eigenvalues: pca.explained_variance_
    #eignevectors: pca.components_
    return scalar, pca

def computeEuclidianDist(X, Y , centroid):
        print(X.shape[0])
        distance = np.zeros(X.shape[0])
        #print(distance)
        #print(centroid.values)
        names = list(centroid.index)
        #print(list(centroid.index))
        #print(Y)
        for y in set(Y):
                tc = centroid.values[names.index(y),:]
                tc = tc.reshape(1,-1)
                distance[Y == y] = metrics.pairwise.pairwise_distances(X[Y == y, :], \
                                                                        tc, metric = 'euclidean').ravel()
        #print(distance)
        return distance
    
def computePCADistance(X,Y, pcas,scalers, pca_labels):
    
    distance = np.zeros(X.shape[0])

    count = 0
    for y in pca_labels:
        
        if X[ Y == y].shape[0] == 0:
            count+=1
            continue
        #scale data first
        scaler = scalers[count]
        
        #acquire pca projection
        x_proj = pcas[count].transform(scaler.transform(X[ Y == y]))
        #compute "PCA distance"
        #equation x^2 / lambda^2
        eigenvalues = pcas[count].explained_variance_

        print (eigenvalues.shape)
        print(x_proj.shape)
        print(np.sum(np.divide(np.square(x_proj), np.square(eigenvalues)), axis = 1).shape)
        distance[Y==y] = np.sum(np.divide(np.square(x_proj), np.square(eigenvalues)), axis = 1)
        count+=1
    #print("Distance: ")
    #print(distance)
    
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
    
def fixedFilter(X,Y, loss, threshold):
    idx_to_keep = np.array([])
    idx_removed = np.array([])
    X_def =[]
    Y_def = []
    class_map = get_class_map()
    for i in range(len(loss)):
        c = Y[i]
        if loss[i]<=threshold[class_map[c]]:
            idx_to_keep=np.append(idx_to_keep,i)
            X_def.append(X[i])
            Y_def.append(Y[i])
        else:
            idx_removed =np.append(idx_removed, i)
    return X_def,Y_def,idx_to_keep, idx_removed

def getTheta(X, Y,distance, percentageToRemove):
    #receive dataset X and Y, sort by the distance and remove the % farthest away from each class
        percentageToKeep = 1 - percentageToRemove
        class_map = get_class_map()
        theta = np.zeros(2)
        for y in set(Y):
            num_to_keep = int(np.round(percentageToKeep * Y[Y==y].shape[0]))
            sorted_distance=np.sort(distance[Y==y])
            theta[class_map[y]] = sorted_distance[num_to_keep - 1 ]
           
        return theta
    
def defense_PCA(datasetName = "spambase",seed = 18, OUTPUT_FOLDER=None):
    train_split_size = 0.7
    validation_size = 0.30
    
    if datasetName == "spambase":
        #load data
        dataset = load_dataset.load_dataset_spambase()
            
        data_size = dataset.shape[0]


        # Prepare data
        array = dataset.values
        X = array[:,0:57]
        Y = array[:,57]
            
        X_train, X_validation, Y_train, Y_validation \
        = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
#       #train_1: clean data, train_2: untrusted data
        X_train_1, X_train_2, Y_train_1, Y_train_2 \
        = model_selection.train_test_split(X_train, Y_train, test_size=train_split_size, random_state=seed)


        mean = dataset.groupby('class').mean()
    

    elif datasetName =="mnist17":
        X_train, Y_train, X_validation, Y_validation = load_dataset.load_dataset_mnist17(sampling=True)
        #train_1: clean data, train_2: untrusted data
        X_train_1, X_train_2, Y_train_1, Y_train_2 \
        = model_selection.train_test_split(X_train, Y_train, test_size=train_split_size, random_state=seed)
        #find median/centroid
        mean_neg = np.mean(X_train[Y_train == -1],axis  =0)
        mean_pos = np.mean(X_train[Y_train == 1], axis = 0)

        mean = pandas.DataFrame(data = [mean_neg, mean_pos], index = [-1,1])
        print(mean_pos.shape)
    
    
    #labels, pcas and scalars's index matches
    labels = set(Y_train_1)
    pcas = []
    scalars = []
    for y in set(Y_train_1):
        #iterate through all labels, perform PCA for each class
        print("Amount of points in class: " + str(y))
        print((X_train_1[Y_train_1 == y]).shape)
        #print(X_train)
        scalar, pca = PCA_analysis(X_train_1[Y_train_1 == y, :])
        pcas.append(pca)
        scalars.append(scalar)
        #also scale all data
        #X_train[Y_train == y] = scalar.transform(X_train[Y_train == y])
        #X_validation[Y_validation == y] = scalar.transform(X_validation[Y_validation == y])
        
    #print(pcas[0].components_)
    print(pcas[0].explained_variance_)

    train_size = X_train.shape[0]
    print("Train size: "+ str(train_size))
    print("Validation size: "+ str(X_validation.shape[0]))

    

    #prepare dataset for model
    train = DataSet(X_train_1, Y_train_1)
    test = DataSet(X_train_1, Y_train_1)
    complete_train = DataSet(X_train, Y_train)
    complete_test = DataSet(X_train, Y_train)
    validation = DataSet(X_validation, Y_validation)
    data_sets = base.Datasets(train = train, validation = validation, test = validation)
    complete_datasets = base.Datasets(train = complete_train, validation = complete_test, test = validation)

    #output directory for attack steps

    output_root = os.path.join(OUTPUT_FOLDER, 'ddd')

    poison_percentage = [0.05,0.1,0.15,0.2]
    step_size = 0.01

    num_points = round(train_size * poison_percentage[3])
    #num_points = 1
    print("Number of Poisoning points: "+str(num_points))
    label_flip = True

    filt = 0.0
    filtList=[]
    accList=[]
    accList_2 = []
    unatt_acc = []
    pca_remove_list = []
    centroid_remove_list = []
    
    for i in range(1):
        model = SmoothHinge(            
                    input_dim=X_train.shape[1],
                    temp=0,
                    weight_decay=0.01,
                    use_bias=True,
                    num_classes=2,
                    batch_size=train_size,
                    data_sets=complete_datasets,
                    initial_learning_rate=0.001,
                    decay_epochs=None,
                    mini_batch=False,
                    train_dir=output_root,
                    log_dir='log',
                    model_name='my_model')

        model.train()
        base_acc = model.get_test_accuracy()

        model.update_train_x_y(X_train_1, Y_train_1)
        model.train()
        
        start_poison = True
        #filter first, then perform the attack for simplicity
        #the order of attack/defense does not matter in game analysis
        applyDefense = True
        if applyDefense == True:
            #to filter, first compute the value of theta using the clean initial dataset
            print("Computing Theta..")
            distance = computePCADistance(X_train_1,Y_train_1,pcas, scalars, labels)
            #distance2 = computeEuclidianDist(poisoned_X,poisoned_Y,centroid=mean)
            
            #theta is an array containing the threshold for each class
            theta = getTheta(X_train_1,Y_train_1,distance, filt)
            print("Value of theta chosen to be: "+str(theta))

            #Then, sanitize the incoming dataset 
            print("Testing Defense...")
            distance2 = computePCADistance(X_train_2,Y_train_2,pcas, scalars, labels)
            
            X_filtered, Y_filtered, indexKept,indexRemoved = fixedFilter(X_train_2,Y_train_2,distance2, theta)           

            print("Filter chosen to be: "+str(filt))
            print("Genuine datapoints removed: "+str(indexRemoved.shape[0]))
            pca_remove_list.append(indexRemoved.shape[0])
            #model.update_train_x_y(X_filtered_2, Y_filtered_2)
            #model.train()

           # acc_2 = model.get_test_accuracy()
            #accList_2.append(acc_2)

            model.update_train_x_y(np.append(X_train_1,X_filtered, axis = 0), np.append(Y_train_1, Y_filtered))
            model.train()
            acc = model.get_test_accuracy()

            #defense without attack
            unatt_acc.append(acc)
            
            X_def = np.append(X_train_1, X_filtered, axis = 0)
            Y_def = np.append(Y_train_1, Y_filtered)

        #start_poison=False
        if start_poison == True:
            #injects into the filtered dataset
            #the attack is aware of the defense, he can pre-calculate the ellisoid filter shape
            #and injects points within filter radius
            X_modified, Y_modified =copy_random_points(
                X_train_2, Y_train_2, 
                target_class=-1, 
                num_copies=num_points, 
                random_seed=seed, 
                replace=True)


            if label_flip:
                Y_modified[X_train_2.shape[0]:] = -Y_modified[X_train_2.shape[0]:]

            #attacker attacks with entire dataset
            X_complete = np.append(X_def, X_modified[X_train_2.shape[0]:], axis = 0)
            Y_complete = np.append(Y_def, Y_modified[X_train_2.shape[0]:])

            model.update_train_x_y(X_complete, Y_complete)
            model.train()
            #acquire projection rule of attack - PCA based
            projection_fn = get_projection_fn(
                X_train_1, Y_train_1,
                pcas, scalars, labels, 
                percentage_to_keep=(1-filt), center = mean
                , num_malicious = num_points
                , sphere = False) 
            #perform the attack
            acc,poisoned_X=iterative_attack.iterative_attack(
                model, 
                indices_to_poison=np.arange(X_def.shape[0], X_complete.shape[0]),            
                test_idx=None, 
                test_description=None, 
                step_size=step_size, 
                num_iter=2000,
                loss_type='normal_loss',
                projection_fn=projection_fn,
                output_root=output_root)
            #acc = model.get_test_accuracy()
            accList.append(acc)
            #print("Shape: "+str(poisoned_X[X_filtered.shape[0]:].shape))
            undef_X = np.concatenate((X_train, poisoned_X[X_def.shape[0]:]), axis=0)
            
            undef_Y = np.concatenate((Y_train, Y_complete[X_def.shape[0]:]), axis=0)
            #print("Shape: "+str(undef_Y.shape))
                        
            model.update_train_x_y(undef_X, undef_Y)
            model.train()
            undef_acc = model.get_test_accuracy()
            accList_2.append(undef_acc)
        #acc = model.get_test_accuracy()
##            acc = newModel.get_test_accuracy()
        filtList.append(filt)
        #accList.append(acc)
        #increment counter, reset model
        filt = filt+0.1
        tf.reset_default_graph()
        #in the end, print result to console
    print(filtList)
    print(unatt_acc)
    print(accList)
    #print(accList_2)
    #print(pca_remove_list)
    #print(centroid_remove_list)
    return filtList, unatt_acc, accList, accList_2, pca_remove_list, centroid_remove_list
        
        
def repeatPCA():
    all_unatt = []
    all_acc = []
    all_acc_undef = []
    all_pca_rm = []
    all_centroid_rm = []
    OUTPUT_FOLDER = '/Users/Yifan/Desktop/Master/output/data'
    result_dir = os.path.join(OUTPUT_FOLDER, "results")
    for i in range(1):
        filtList, unatt_acc, accList, accList_2, pca_remove_list, centroid_remove_list = defense_PCA(datasetName = "mnist17", seed = random.randint(1,100),OUTPUT_FOLDER=OUTPUT_FOLDER)
        all_unatt.append(unatt_acc)
        all_acc.append(accList)
        all_acc_undef.append(accList_2)
        all_pca_rm.append(pca_remove_list)
        all_centroid_rm.append(centroid_remove_list)
    print(filtList) #the filter radius being tested
    print(all_unatt) #all defense without attack model accuracy
    print(all_acc) #all defended model accuracy
    print(all_acc_undef) #all udefended model accuracy
#    print(all_acc_center) #not in use
#    print(all_pca_rm) 
#    print(all_centroid_rm)
    all_results = [["Filter percentiles", filtList], ["Accuracy Defence Without Attack", all_unatt],
                   ["Accuracy Attack with PCA Defence", all_acc], ["Accuracy of Attack Without Defence", all_acc_undef],
                    ["Number of Genuine Points Removed: ", all_pca_rm]
                   ]
    write_result.write_result(all_results, result_dir, "PCA_Game.txt")

repeatPCA()


    

    
