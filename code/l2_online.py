from influence.influence.smooth_hinge import SmoothHinge
from influence.influence.dataset import DataSet
from influence.influence.SimpleNeuralNet import SimpleNeuralNet
import load_dataset

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base

# Load libraries
import pandas
import numpy as np
#import autograd
#from pandas.plotting import scatter_matrix
#import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import defenses
import iterative_attack
import random
import write_result


from tensorflow.examples.tutorials.mnist import input_data

import os


def computeEuclidianDist(X, Y , centroid):
    #computes euclidian distance of all datapoints from centroid
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


def get_projection_fn(
    X_clean, 
    Y_clean, 
    sphere=True,
    slab=False,
    percentile=70):

    class_map, centroids, centroid_vec, sphere_radii, slab_radii= get_data_params(X_clean, Y_clean, percentile)
    #print("Radii:")
    #print(sphere_radii)
    def project_onto_feasible_set(X, Y):
        if sphere:
            X = project_onto_sphere(X, Y, sphere_radii, centroids, class_map)

        elif slab:
            X = project_onto_slab(X, Y, centroid_vec, slab_radii, centroids, class_map)
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

def get_centroids(X, Y, class_map):
    num_classes = len(set(Y))
    num_features = X.shape[1]
    centroids = np.zeros((num_classes, num_features))
    for y in set(Y):            
        centroids[class_map[y], :] = np.median(X[Y == y, :], axis=0)
    return centroids

def get_centroid_vec(centroids):
    assert centroids.shape[0] == 2
    centroid_vec = centroids[0, :] - centroids[1, :]
    centroid_vec /= np.linalg.norm(centroid_vec)
    centroid_vec = np.reshape(centroid_vec, (1, -1))
    return centroid_vec


def attackWithDefense(datasetName = "spambase", seed = 18, OUTPUT_FOLDER=None):
        
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


        median = dataset.groupby('class').median()
    

    elif datasetName =="mnist17":
        X_train, Y_train, X_validation, Y_validation = load_dataset.load_dataset_mnist17(sampling=True)
        #train_1: clean data, train_2: untrusted data
        X_train_1, X_train_2, Y_train_1, Y_train_2 \
        = model_selection.train_test_split(X_train, Y_train, test_size=train_split_size, random_state=seed)
        #find median/centroid
        median_neg = np.median(X_train[Y_train == -1],axis  =0)
        median_pos = np.median(X_train[Y_train == 1], axis = 0)

        median = pandas.DataFrame(data = [median_neg, median_pos], index = [-1,1])
        print(median_pos.shape)
    
    print("Data size:"+str(X_train.shape[0])+","+str(X_validation.shape[0]))

    train_size = X_train.shape[0]
    print(train_size)


    ##with tf.Session() as sess:
    ##    Y_train = tf.one_hot(Y_train,2).eval()
    ##    Y_validation = tf.one_hot(Y_validation,2).eval()
    ##print(X_validation.shape)
    ##print(Y_validation.shape)

    #prepare dataset for model
    train = DataSet(X_train_1, Y_train_1)
    test = DataSet(X_train_1, Y_train_1)
    complete_train = DataSet(X_train, Y_train)
    complete_test = DataSet(X_train, Y_train)
    validation = DataSet(X_validation, Y_validation)
    
    data_sets = base.Datasets(train = train, validation = validation, test = validation)
    complete_datasets = base.Datasets(train = complete_train, validation = complete_test, test = validation)
    #the output directory of attack steps
    output_root = os.path.join(OUTPUT_FOLDER, 'ddd')

    poison_percentage = [0.05,0.1,0.15,0.2]
    step_size = 0.01

    #filter values
    filt = 0.0
    #defense_filts = [50,30,10,5,3,1, 0.1]
    #defense_filts = [30,10,5,3,1, 0.5]
    #defense_filts = [0.5]
    epsilon = 0.1
    
    filtList = []
    accList = []
    accList_2 = []
    unatt_acc = []
    l2_remove_list=[]
    num_points = round(train_size * poison_percentage[3])
    print("Number of Poisoning points: "+str(num_points))
    for i in range(8):

       
        label_flip = True



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

        applyDefense = True
        if applyDefense == True:
            #to filter, first compute the value of theta using the clean initial dataset
            print("Computing Theta..")
            distance = computeEuclidianDist(X_train_1,Y_train_1,centroid=median)
            #distance2 = computeEuclidianDist(poisoned_X,poisoned_Y,centroid=mean)
            
            #theta is an array containing the threshold for each class
            theta = getTheta(X_train_1,Y_train_1,distance, filt)
            print("Value of theta chosen to be: "+str(theta))

            #Then, sanitize the incoming dataset 
            print("Testing Defense...")
            distance2 = computeEuclidianDist(X_train_2,Y_train_2,centroid=median)
            
            X_filtered, Y_filtered, indexKept,indexRemoved = fixedFilter(X_train_2,Y_train_2,distance2, theta)           

            print("Filter chosen to be: "+str(filt))
            print("Genuine datapoints removed: "+str(indexRemoved.shape[0]))
            l2_remove_list.append(indexRemoved.shape[0])
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
        
        start_poison = True
        if start_poison == True:

            #injects positive class data (invert the label in next step)
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

            print(X_complete.shape)
            print(Y_complete.shape)
            model.update_train_x_y(X_complete, Y_complete)
            model.train()


            #acquire projection rules for attack
            projection_fn = get_projection_fn(
                X_train_1, Y_train_1,
                sphere=True,
                slab=False,
                percentile=round(100-filt*100))
            
            #perform the attack
            min_acc, poisoned_X = iterative_attack.iterative_attack(
                model, 
                indices_to_poison=np.arange(X_def.shape[0], X_complete.shape[0]),            
                test_idx=None, 
                test_description=None, 
                step_size=step_size, 
                num_iter=2000,
                loss_type='normal_loss',
                projection_fn=projection_fn,
                output_root=output_root)
            #print(model.data_sets.train.x)

        #acc = model.get_test_accuracy()
            accList.append(min_acc)
            #print("Shape: "+str(poisoned_X[X_filtered.shape[0]:].shape))
            undef_X = np.concatenate((X_train, poisoned_X[X_def.shape[0]:]), axis=0)
            
            undef_Y = np.concatenate((Y_train, Y_complete[X_def.shape[0]:]), axis=0)
            #print("Shape: "+str(undef_Y.shape))
                        
            model.update_train_x_y(undef_X, undef_Y)
            model.train()
            undef_acc = model.get_test_accuracy()
            accList_2.append(undef_acc)
            
            #acc = model.get_test_accuracy()
            #record accuracy into accList
            filtList.append(filt)
            #accList.append(acc)
            #increment counter, reset model
            filt = filt+0.1
            tf.reset_default_graph()
    #print result to the console for now
    print(filtList)
    print(accList)
    return filtList, unatt_acc, accList, accList_2, l2_remove_list

def repeat():
    all_acc_list = []
    all_unatt = []
    all_acc_undef = []
    all_l2_rm = []
    OUTPUT_FOLDER = '/Users/Yifan/Desktop/Master/output/data'
    result_dir = os.path.join(OUTPUT_FOLDER, "results")
    for i in range(1):
        filtList, unatt_acc, accList, accList_2, l2_remove_list = attackWithDefense(datasetName = "spambase", seed = random.randint(1,100),OUTPUT_FOLDER = OUTPUT_FOLDER)
        all_unatt.append(unatt_acc)
        all_acc_list.append(accList)
        all_acc_undef.append(accList_2)
        all_l2_rm.append(l2_remove_list)
    print(filtList)
    print(all_acc_list)
    all_results = [["Filter percentiles", filtList], ["Accuracy Defence Without Attack", all_unatt],
                   ["Accuracy Attack with Centroid Defence", all_acc_list], ["Accuracy of Attack Without Defence", all_acc_undef],
                    ["Number of Genuine Points Removed: ", all_l2_rm]
                   ]
    write_result.write_result(all_results, result_dir, "Attack_with_defense.txt")

repeat()
