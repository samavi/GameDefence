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
from sklearn.cluster import DBSCAN
import defenses
import iterative_attack
import random
import write_result
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

from tensorflow.examples.tutorials.mnist import input_data

import os
import math

def computeEuclidianDist(X, Y , centroid):
    #computes euclidian distance of all datapoints from centroid
        #print(X.shape[0])
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

def computeEuclidianDistForOne(X, point):
    #print(X.shape)
    distance = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        distance[i,:] = (X[i]-point) * (X[i]-point) 
    
    return distance
def fixedFilter(X,Y, loss, threshold):
    #filter with fixed threshold value
    idx_to_keep = []
    idx_removed = []
    X_def =[]
    Y_def = []
    for i in range(len(loss)):
        if loss[i]<=threshold:
            idx_to_keep.append(i)
            X_def.append(X[i])
            Y_def.append(Y[i])
        else:
            idx_removed.append(i)
    return X_def,Y_def,idx_to_keep, idx_removed
def filterData(X, Y, distance, percentageToRemove):
        #receive dataset X and Y, sort by the distance and remove the % farthest away
        percentageToKeep = 1 - percentageToRemove

        idx_to_keep = []
        filt_value = []
        #print(type(idx_to_keep))
        for y in set(Y): 
            num_to_keep = int(np.round(percentageToKeep * np.sum(Y == y)))
            sorted_indices = np.argsort(distance[Y == y])
            sorted_distance=np.sort(distance[Y == y])
            print(sorted_distance)
            filt_value.append(sorted_distance[num_to_keep - 1 ])
            
            idx_to_keep.append( \
                    np.where(Y == y)[0][sorted_indices[:num_to_keep]])

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
        return X_def, Y_def, idx_to_keep, filt_value
    
def filterData_Curie(X, Y,distance, percentageToRemove):
    #Curie disregards the labels(Y) at this stage
    X_def,_,idx_to_keep, filt_value=filterData(X,np.ones(X.shape[0]),distance, percentageToRemove)
    
    #Recompute the correct Y to return
    mask = np.zeros(X.shape[0])
    for i in idx_to_keep:
        mask[i] = 1
        
    return X_def, Y[mask==1], idx_to_keep, filt_value
    

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
    clusters = None,
    distance = None,
    sphere=True,
    slab=False,
    omega=1,
    theta=None,
    target_class=None
    ):

    class_map, centroids, centroid_vec, sphere_radii= get_data_params_Curie(X_clean, Y_clean, clusters,distance, omega,theta,target_class)
    if centroids is None:
            return None
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

def get_data_params_Curie(X, Y, clusters,distance, omega, theta, target_class):
    #The centroid has the closest distance^2 to all other points in the cluster
    #Therefore we can measure the distance of poisoning points to the centroid, and apply projection based on the recalculated dist
    num_classes = len(set(Y))
    num_features = X.shape[1]
    centroids = np.zeros((num_classes, num_features))
    class_map = get_class_map()
    centroids, sphere_radii = get_centroidRadius_Curie(X, Y, clusters,distance, class_map, omega, theta, target_class)
    if centroids is None:
            return None, None,None,None
##    # Get radii for Curie sphere
##    sphere_radii = np.zeros(2)
##    dists = defenses.compute_dists_under_Q(    
##        X, Y,
##        Q=None,
##        centroids=centroids,
##        class_map=class_map,    
##        norm=2)
##    for y in set(Y):            
##        sphere_radii[class_map[y]] = np.percentile(dists[Y == y], percentile)

    # Get vector between centroids
    centroid_vec = get_centroid_vec(centroids)

##    # Get radii for slab
##    slab_radii = np.zeros(2)
##    for y in set(Y):            
##        dists = np.abs( 
##            (X[Y == y, :].dot(centroid_vec.T) - centroids[class_map[y], :].dot(centroid_vec.T)))            
##        slab_radii[class_map[y]] = np.percentile(dists, percentile)

    return class_map, centroids, centroid_vec, sphere_radii


def get_centroids(X, Y, class_map):
    num_classes = len(set(Y))
    num_features = X.shape[1]
    centroids = np.zeros((num_classes, num_features))
    for y in set(Y):            
        centroids[class_map[y], :] = np.mean(X[Y == y, :], axis=0)
    return centroids

def get_centroidRadius_Curie(X, Y, clusters,distance, class_map, omega, theta, target_class):
    num_classes = len(set(Y))
    num_features = X.shape[1]
    centroids = np.zeros((num_classes, num_features))

    sphere_radii = np.zeros(2)

    all_average_dist = distance
    for y in set(Y):
        
        #find the largest cluster in each class and compute its center
        c = np.bincount(clusters[(Y==y) & (clusters>=0) ]).argmax()

        print( "C: "+ str(c) )
        print("Amount: " + str(clusters[(Y==y) & (clusters == c)].shape[0]))
        centroids[class_map[y], :] = np.mean(X[(Y == y) & (clusters == c ), :], axis=0)

        # Then, compute radius
        #each malicious point incurs 2*omega^2/ loss, subtract from budget
        budget = theta - 2*(omega**2)
        print("Budget: "+ str(budget))
        #assert budget >= 0, "Budget is less than 0, attack not possible"
        if budget < 0 and (target_class is None or y==target_class):
                return None, None

        #The remaining is the movement budget around the centroid, after some calculation
        #k = np.zeros(num_features)
        #for point in X[(Y==y) & (clusters==c)]:
        #    k = k + (point - centroids[class_map[y], :])
        #print(k)
        sphere_radii[class_map[y]] = math.sqrt(budget)
    print("Radii: "+ str(sphere_radii))
        

    #filp the centroids (+1 datapoints are constrainted by the -1 centroid, and vice versa)
    centroids = centroids[::-1]
        
    return centroids, sphere_radii


def get_centroid_vec(centroids):
    assert centroids.shape[0] == 2
    centroid_vec = centroids[0, :] - centroids[1, :]
    centroid_vec /= np.linalg.norm(centroid_vec)
    centroid_vec = np.reshape(centroid_vec, (1, -1))
    return centroid_vec

    
def plotEps(X):
    neigh = NearestNeighbors(n_neighbors = 2)

    nbrs = neigh.fit(X)
    distance, indices = nbrs.kneighbors(X)

    distance = np.sort(distance, axis=0)
    distance = distance[:,1]
    plt.plot(distance)
    plt.show()

def clusterData(X, Y,eps, plotName = "plot.png"):
    #accroding to their psudocode, perform PCA on dataset first
        pca=PCA(n_components = 2)
        pca_X = pca.fit_transform(X)

        #find out best value for eps
        #plotEps(pca_X)
        
        #cluster the data first
        clustering = DBSCAN(eps=eps).fit(pca_X)

        clusters = clustering.labels_

        colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy']
        vectorizer = np.vectorize(lambda x: colors[x % len(colors)])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(xs=pca_X[:,0], ys=pca_X[:,1],zs=Y, c=vectorizer(clusters))

        #uncomment me to display the clustering result
        #plt.show()
        plt.savefig(plotName)

        return clustering

def computeOmega(X, Y, clusters, contribPercentage):
        print("Computing Omega..")
        print("Omega contributes to "+str(100*contribPercentage) + "percent of Curie distance.")
        all_average_dist = np.array([])
        for point_index in range(Y.shape[0]):
            #calculate pairwise distance with points in the same cluster
            cls = clusters[point_index]
            #print(cls)
            dist = computeEuclidianDistForOne(X[clusters==cls,:] , X[point_index])
            average_dist = np.sum(dist)/X[clusters==cls,:].shape[0]
            all_average_dist = np.append(all_average_dist, average_dist)
        #find the maximum intra-cluster distance
        max_dist = all_average_dist.max()
        #set omega^2 to max_dist/(1-contribPercentage) * contribPercentage
        #divide it by 2, because labels are mapped to 1 and -1 (instead of 1 and 0)
        return math.sqrt(max_dist/(1-contribPercentage) * contribPercentage)/2

def computeCurieDistance(X, Y, clusters, omega):
    #append label to feature values, in prep for Curie distance calculation
        F = np.concatenate( (X, omega * Y[:,None]), axis=1)
        print("Shape of Feat+Label:" + str(F.shape))
        print(F)

        print("Computing Distance..")
        all_average_dist = np.array([])
        for point_index in range(Y.shape[0]):
            #calculate pairwise distance with points in the same cluster
            cls = clusters[point_index]
            #print(cls)
            dist = computeEuclidianDistForOne(F[clusters==cls,:] , F[point_index])
            average_dist = np.sum(dist)/F[clusters==cls,:].shape[0]
            all_average_dist = np.append(all_average_dist, average_dist)
        
        #convert distance to its z-score
        #all_average_dist = stats.zscore(all_average_dist)
        
        #print(all_average_dist)
        return all_average_dist

def getTheta(X, Y,distance, percentageToRemove):
    #receive dataset X and Y, sort by the distance and remove the % farthest away
        percentageToKeep = 1 - percentageToRemove


        num_to_keep = int(np.round(percentageToKeep * Y.shape[0]))
        sorted_distance=np.sort(distance)
        print(sorted_distance)
           
        return sorted_distance[num_to_keep - 1 ]

def attackWithDefense(datasetName = "spambase", seed = 18, OUTPUT_FOLDER=None, PLOT_FOLDER=None):
        
    validation_size = 0.3
    train_split_size = 0.7
    if datasetName == "spambase":
        #load data
        dataset = load_dataset.load_dataset_spambase()
            
        data_size = dataset.shape[0]


        # Prepare data
        array = dataset.values
        X = array[:,0:57]
        Y = array[:,57]

        # Split-out validation dataset
        X_train, X_validation, Y_train, Y_validation \
        = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


        
        #find median/centroid
        median = dataset.groupby('class').median()
        #pre-determined value using plotEps(), used when clustering
        eps = 60
    elif datasetName =="mnist17":
        X_train, Y_train, X_validation, Y_validation = load_dataset.load_dataset_mnist17(sampling=True)
        #find median/centroid
        median_neg = np.median(X_train[Y_train == -1],axis  =0)
        median_pos = np.median(X_train[Y_train == 1], axis = 0)

        median = pandas.DataFrame(data = [median_neg, median_pos], index = [-1,1])
        print(median_pos.shape)
        
        #pre-determined value using plotEps(), used when clustering
        eps = 40
    elif datasetName =="mnist01":
        X_train, Y_train, X_validation, Y_validation = load_dataset.load_dataset_mnist01(sampling=True)
        
        #find median/centroid
        median_neg = np.median(X_train[Y_train == -1],axis  =0)
        median_pos = np.median(X_train[Y_train == 1], axis = 0)

        median = pandas.DataFrame(data = [median_neg, median_pos], index = [-1,1])
        print(median_pos.shape)
        
    print("Data size:"+str(X_train.shape[0])+","+str(X_validation.shape[0]))

    train_size = X_train.shape[0]
    print(train_size)

    days = 1
    incoming_sets_X = []
    incoming_sets_Y = []
    #separate data into one large initial clean set, and 10 incoming dataset
    #train_1: clean data, train_2: untrusted data
    X_train_1, X_train_2, Y_train_1, Y_train_2 \
    = model_selection.train_test_split(X_train, Y_train, test_size=train_split_size, random_state=seed)

    num_split = math.floor(X_train_2.shape[0]/10)
    for d in range(days-1):
        incoming_sets_X.append(X_train_2[d*num_split: (d+1)*num_split,:])
        incoming_sets_Y.append(Y_train_2[d*num_split: (d+1)*num_split])
    #append the remaining points into the last set
    incoming_sets_X.append(X_train_2[(days-1)*num_split:X_train_2.shape[0],:])
    incoming_sets_Y.append(Y_train_2[(days-1)*num_split:X_train_2.shape[0]])


    ##with tf.Session() as sess:
    ##    Y_train = tf.one_hot(Y_train,2).eval()
    ##    Y_validation = tf.one_hot(Y_validation,2).eval()
    ##print(X_validation.shape)
    ##print(Y_validation.shape)


    #output directory for attack steps
    
    output_root = os.path.join(OUTPUT_FOLDER, 'ddd')

    poison_percentage = [0.05,0.1,0.15,0.2]
    step_size = 0.01

    #filter values
    filt = 0.073
    omegaContribPercentage = 0.8
    
    filtList = []
    thetaList = []
    accList = []
    undefList = []
    datasetList_X = []
    datasetList_Y = []
    removeList = []
    attackList=[]
    base_acc = 0
    
    num_points_total = round(train_size * poison_percentage[1])

    #determine value of theta using the clean dataset (without removing any points at this stage)
    
    #cluster the data first
    clustering = clusterData(X_train_1, Y_train_1, eps, plotName = PLOT_FOLDER + "/" + datasetName + "_orignal.png")

    clusters = clustering.labels_

    #compute omega
    omega = computeOmega(X_train_1, Y_train_1, clusters, omegaContribPercentage)
    
    all_average_dist = computeCurieDistance(X_train_1, Y_train_1, clusters, omega )

    #theta is a single value for the entire dataset (instead of 1 per class)
    theta = getTheta(X_train_1,Y_train_1,all_average_dist, filt)
    print("Value of theta chosen to be: "+str(theta))

    #num_points = 1
    print("Number of Total Poisoning points: "+str(num_points_total))

    #each day, the attacker injects (total/day) amount of malicious points (up to 20% at the end)
    num_points_each_day = int(num_points_total/days)

    #initialize X/Y_train_new, for iterations
    X_train_new = X_train_1
    Y_train_new = Y_train_1
    for i in range(days):
        
        
        label_flip = True

        #prepare dataset for model
        train = DataSet(X_train_new, Y_train_new)
        test = DataSet(X_validation, Y_validation)
        validation = DataSet(X_validation, Y_validation)
        data_sets = base.Datasets(train = train, validation = validation, test = test)


        #dataset changes every day
        model = SmoothHinge(            
                    input_dim=X_train.shape[1],
                    temp=0,
                    weight_decay=0.01,
                    use_bias=True,
                    num_classes=2,
                    batch_size=X_train_new.shape[0],
                    data_sets=data_sets,
                    initial_learning_rate=0.001,
                    decay_epochs=None,
                    mini_batch=False,
                    train_dir=output_root,
                    log_dir='log',
                    model_name='my_model')




        model.train()

        if base_acc ==0:
                base_acc = model.get_test_accuracy()
        #the defender starts by filtering
        #attacker attacks with entire dataset
        X_clean = np.append(X_train_new, incoming_sets_X[i], axis = 0)
        Y_clean = np.append(Y_train_new, incoming_sets_Y[i])
            
        clustering = clusterData(X_clean, Y_clean, eps, plotName = PLOT_FOLDER + "/" + datasetName + "_ahead_filtered_day"+str(i)+".png")

        pre_clusters = clustering.labels_
    
        all_average_dist = computeCurieDistance(X_clean, Y_clean, pre_clusters, omega )

        #only filter data from the incoming set
        X_def,Y_def,indexKept, idx_removed = fixedFilter(X_clean[X_train_new.shape[0]:,:], Y_clean[X_train_new.shape[0]:], all_average_dist[X_train_new.shape[0]:], theta)

        removeList.append(len(idx_removed))
        X_def = np.append(X_train_new, X_def, axis = 0)
        Y_def = np.append(Y_train_new, Y_def)
        print(X_def.shape)
        print(Y_def.shape)
        #attacker use the predicted filter result to optimize his attack
        start_poison = True
        if start_poison == True:
            #injects positive class datapoints(invert the label in next step)
            target_class = 1
            
            X_modified, Y_modified =copy_random_points(
                incoming_sets_X[i], incoming_sets_Y[i], 
                target_class=-1, 
                num_copies=num_points_each_day, 
                random_seed=seed, 
                replace=True)

            ##X_modified, Y_modified = copy_random_points(
            ##    X_modified, Y_modified, 
            ##    target_class=-1, 
            ##    num_copies=num_neg_copies, 
            ##    random_seed=random_seed, 
            ##    replace=True)

            if label_flip:
                Y_modified[incoming_sets_X[i].shape[0]:] = -Y_modified[incoming_sets_X[i].shape[0]:]

            #print(X_modified)
                
            #attacker attacks with entire dataset
            X_complete = np.append(X_def, X_modified[X_train_2.shape[0]:], axis = 0)
            Y_complete = np.append(Y_def, Y_modified[X_train_2.shape[0]:])
            
            model.update_train_x_y(X_complete, Y_complete)
            model.train()

            #acquire projection rules for attack
            projection_fn = get_projection_fn(
                X_train_new, Y_train_new, clusters = clusters,
                distance = all_average_dist,
                sphere=True,
                slab=False,
                omega=omega,
                theta = theta,
                target_class = target_class)

            if projection_fn is not None:
                    #perform the attack
                    min_acc, min_X = iterative_attack.iterative_attack(
                        model, 
                        indices_to_poison=np.arange(X_def.shape[0], X_complete.shape[0]),            
                        test_idx=None, 
                        test_description=None, 
                        step_size=step_size, 
                        num_iter=2000,
                        loss_type='normal_loss',
                        projection_fn=projection_fn,
                        output_root=output_root)
                    attackList.append("Y")
                    #before proceeding, measure accuracy without sanitization
                    X_no_def = np.append(X_clean, min_X[X_def.shape[0]:,:], axis = 0)
                    Y_no_def = np.append(Y_clean, Y_complete[X_def.shape[0]:])
            else:
                    min_X = X_def
                    Y_complete = Y_def
                    model.update_train_x_y(X_def, Y_def)
                    model.train()
                    min_acc = model.get_test_accuracy()
                    attackList.append("N")
                    #before proceeding, measure accuracy without sanitization
                    X_no_def = X_clean
                    Y_no_def = Y_clean

            #print(model.data_sets.train.x)
            print(min_X.shape)




            model.update_train_x_y(X_no_def, Y_no_def)
            model.train()
            
            acc = model.get_test_accuracy()

            #prepare dataset for next day
            X_train_new = min_X
            Y_train_new = Y_complete
            
            #record accuracy into accList
            filtList.append(i)
            thetaList.append(theta)
            accList.append(min_acc)
            undefList.append(acc)
            datasetList_X.append(X_train_new)
            datasetList_Y.append(Y_train_new)
            
            #increment counter, reset model
            #filt = filt+0.1
            tf.reset_default_graph()
            
            #re-cluster and compute distance
            #cluster the data first
            clustering = clusterData(X_train_new, Y_train_new, eps, plotName = PLOT_FOLDER + "/" + datasetName + "_poisoned_day"+str(i)+".png")

            clusters = clustering.labels_
    
            all_average_dist = computeCurieDistance(X_train_new, Y_train_new, clusters, omega )

            #theta is a single value for the entire dataset (instead of 1 per class)
            theta = getTheta(X_train_new,Y_train_new,all_average_dist, filt)
           
            print("Value of theta chosen to be: "+str(theta))
    #print result to the console for now
    print(filtList)
    print(thetaList)
    print(undefList)
    print(accList)
    return filtList, thetaList, accList, undefList, datasetList_X, datasetList_Y, removeList, base_acc, attackList

def repeat():
    all_acc_list = []
    all_undef_list = []
    OUTPUT_FOLDER = '/Users/Yifan/Desktop/Master/output/data'
    PLOT_FOLDER = '/Users/Yifan/Desktop/Master/output/plots'
    result_dir = os.path.join(OUTPUT_FOLDER, "results")
    for i in range(1):
        filtList, thetaList, accList, undefList, datasetList_X, datasetList_Y, removeList, base_acc, attackList = attackWithDefense(datasetName = "mnist17", seed = 18,OUTPUT_FOLDER = OUTPUT_FOLDER, PLOT_FOLDER=PLOT_FOLDER)
        all_acc_list.append(accList)
        all_undef_list.append(undefList)
    print(filtList)
    print(all_acc_list)
    all_results = [["Day", filtList], ["Theta", thetaList ],["Accuracy with Defence", all_acc_list]
                   , ["Accuracy without Defence", all_undef_list], ["Genuine Points Removed:", removeList]
                   , ["Day 0 Accuracy: ", [base_acc] ], ["Attacked: ", attackList] ]
    write_result.write_result(all_results, result_dir, "Multiday_Curie.txt")
    #Also, write the intemediate datasets to a file (for loading convenience)
    write_intemediate = True
    if write_intemediate:
        for i in range(len(datasetList_X)):
            data = [datasetList_X, datasetList_Y]
            write_result.write_result(all_results, result_dir, "Multiday_Curie_"+str(i)+".dat")

repeat()
