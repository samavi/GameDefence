from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import abc
import sys

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, cluster, svm
import matplotlib.pyplot as plt
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse 
from scipy.optimize import fmin_l_bfgs_b, fmin_cg, fmin_ncg

import os.path
import time
import tensorflow as tf
import math

from influence.influence.genericNeuralNet import GenericNeuralNet, variable, variable_with_weight_decay

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def scale(scaler, X):
    mean = scaler.mean_
    var = scaler.var_

    return (X - mean)/var

def pca_transform(pca, X):
    eigenvectors = pca.components_

    inv_eigen = tf.linalg.inv(eigenvectors)

    new_coord = tf.multiply(tf.cast(X, tf.float64), inv_eigen)

    return new_coord

def computePCALocation(X,Y, pcas,scalers, pca_labels, radii, elevation = 1):
    
    distance = tf.zeros_like(Y, dtype = tf.float64)
    inbound = tf.zeros_like(Y, dtype = tf.float32)

    count = 0
    for y in pca_labels:
        
        if tf.boolean_mask(X, tf.equal(Y,y)).shape[0] == 0:
            count+=1
            continue
        
        radius = radii[count]
        #scale data first
        scaler = scalers[count]

        pca = pcas[count]
        
        #acquire pca projection
        x_proj = tf.multiply(
        pca_transform(pca,
            scale(scaler, X)
                )
            
        , tf.cast(tf.equal(Y,y),tf.float64)
        )
                
            
        #compute "PCA distance"
        #equation x^2 / lambda^2
        eigenvalues = pcas[count].explained_variance_

        print (eigenvalues.shape)
        print(x_proj.shape)
        print(tf.reduce_sum(tf.divide(tf.square(x_proj), tf.square(eigenvalues)), axis = 1))
        distance = tf.add(distance,
            tf.reduce_sum(tf.divide(tf.square(x_proj), tf.square(eigenvalues)), axis = 1)
        )
        
        count+=1
    #print("Distance: ")
    #print(distance)
        
    #inbound[(Y==y) & (distance <= radius)] = elevation
    inboud = tf.add(inbound,
                    tf.multiply(tf.cast(elevation, tf.float32),
                tf.cast(tf.less_equal(distance, radius), tf.float32)
                                )
        )
    return inbound

    
def log_loss(x, t):
    exponents = -(x-1)/t
    # exponents = -(x)/t
    max_elems = tf.maximum(exponents, tf.zeros_like(exponents))

    return t * (max_elems + tf.log(
        tf.exp(exponents - max_elems) + 
        tf.exp(tf.zeros_like(exponents) - max_elems)))
    # return t * tf.log(tf.exp(-(x)/t) + 1)        

def hinge(x):
    return tf.maximum(1-x, 0)

def smooth_hinge_loss(x, t):    

    # return tf.cond(
    #     tf.equal(t, 0),
    #     lambda: hinge(x),
    #     lambda: log_loss(x,t)
    #     )

    if t == 0:
        return hinge(x)
    else:
        return log_loss(x,t)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    a = sigmoid(x)
    return a * (1 - a)


class SmoothHingeElevated(GenericNeuralNet):

    # Expects labels to be +1 or -1

    def __init__(self, input_dim, temp, weight_decay, use_bias,
                 pcas, scalers, radii, pca_labels,**kwargs):
        self.weight_decay = weight_decay
        self.input_dim = input_dim
        self.temp = temp
        self.use_bias = use_bias

        self.pcas = pcas
        self.scalers = scalers
        self.radii = radii
        self.pca_labels = pca_labels

        self.elevation = 1000000

        super(SmoothHingeElevated, self).__init__(**kwargs)

        C = 1.0 / (self.num_train_examples * self.weight_decay)        
        self.svm_model = svm.LinearSVC(
            C=C,
            loss='hinge',
            tol=1e-6,
            fit_intercept=self.use_bias,
            random_state=24,
            max_iter=5000)

        C_minus_one = 1.0 / ((self.num_train_examples - 1) * self.weight_decay)
        self.svm_model_minus_one = svm.LinearSVC(
            C=C_minus_one,
            loss='hinge',
            tol=1e-6,
            fit_intercept=self.use_bias,
            random_state=24,
            max_iter=5000)     

        self.set_params_op = self.set_params()

        self.W = np.zeros(input_dim+1)

        assert self.num_classes == 2

    def get_all_params(self):
        all_params = []
        for layer in ['softmax_linear']:
            # for var_name in ['weights', 'biases']:
            for var_name in ['weights']:
                temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))            
                all_params.append(temp_tensor)      
        return all_params        
        

    def placeholder_inputs(self):
        input_placeholder = tf.placeholder(
            tf.float32, 
            shape=(None, self.input_dim),
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.int32,             
            shape=([None]),
            name='labels_placeholder')
        return input_placeholder, labels_placeholder


    def inference(self, input):        
        # Softmax_linear
        with tf.variable_scope('softmax_linear'):

            # We regularize the bias to keep it in line with sklearn's 
            # liblinear implementation
            if self.use_bias: 
                weights = variable_with_weight_decay(
                    'weights', 
                    [self.input_dim + 1],
                    stddev=5.0 / math.sqrt(float(self.input_dim)),
                    wd=self.weight_decay)            
                # biases = variable(
                #     'biases',
                #     [1],
                #     tf.constant_initializer(0.0))
                
                #bias is already accounted as the last weight value
                #just pad the multiplier with 1 to properly add it
                logits = tf.matmul(
                    tf.concat([input, tf.ones([tf.shape(input)[0], 1])], axis=1),
                    tf.reshape(weights, [-1, 1]))# + biases
            
            else: 
                weights = variable_with_weight_decay(
                    'weights', 
                    [self.input_dim],
                    stddev=5.0 / math.sqrt(float(self.input_dim)),
                    wd=self.weight_decay)            

                logits = tf.matmul(
                    input,
                    tf.reshape(weights, [-1, 1]))


        self.weights = weights
        return logits


    def retrain(self, num_steps, feed_dict):
        # self.sess.run(
        #     self.update_learning_rate_op, 
        #     feed_dict={self.learning_rate_placeholder: 1 * self.initial_learning_rate})        

        # for step in xrange(num_steps):   
        #     self.sess.run(self.train_op, feed_dict=feed_dict)
        if self.temp == 0:
            self.train_with_svm(feed_dict, save_checkpoints=False, verbose=False)
        else:
            self.train_with_fmin(feed_dict, save_checkpoints=False, verbose=False)

    def get_train_fmin_loss_fn(self, train_feed_dict):
        def fmin_loss(W):
            params_feed_dict = {}
            params_feed_dict[self.W_placeholder] = W        
            self.sess.run(self.set_params_op, feed_dict=params_feed_dict)
            loss_val = self.sess.run(self.total_loss, feed_dict=train_feed_dict)        
            return loss_val
        return fmin_loss

    def get_train_fmin_grad_fn(self, train_feed_dict):        
        def fmin_grad(W):
            params_feed_dict = {}
            params_feed_dict[self.W_placeholder] = W        
            self.sess.run(self.set_params_op, feed_dict=params_feed_dict)
            grad_val = self.sess.run(self.grad_total_loss_op, feed_dict=train_feed_dict)[0]
            return grad_val
        return fmin_grad


    def get_train_fmin_hvp_fn(self, train_feed_dict):
        def fmin_hvp(W, v):            
            params_feed_dict = {}
            params_feed_dict[self.W_placeholder] = W        
            self.sess.run(self.set_params_op, feed_dict=params_feed_dict)

            feed_dict = self.update_feed_dict_with_v_placeholder(train_feed_dict, self.vec_to_list(v))
            hessian_vector_val = self.sess.run(self.hessian_vector, feed_dict=feed_dict)[0]            
            return hessian_vector_val
        return fmin_hvp


    def train(self):
        if self.temp == 0:
            self.train_with_svm(self.all_train_feed_dict)
        else:
            self.train_with_fmin(self.all_train_feed_dict)
            
    def train_with_fmin(self, train_feed_dict, save_checkpoints=True, verbose=True):
        fmin_loss_fn = self.get_train_fmin_loss_fn(train_feed_dict)
        fmin_grad_fn = self.get_train_fmin_grad_fn(train_feed_dict)
        fmin_hvp_fn = self.get_train_fmin_hvp_fn(train_feed_dict)

        x0 = np.array(self.sess.run(self.params)[0])
        
        # fmin_results = fmin_l_bfgs_b(
        # # fmin_results = fmin_cg(
        #     fmin_loss_fn,
        #     x0,
        #     fmin_grad_fn
        #     # gtol=1e-8
        #     )

        fmin_results = fmin_ncg(
            f=fmin_loss_fn,
            x0=x0,
            fprime=fmin_grad_fn,
            fhess_p=fmin_hvp_fn,            
            avextol=1e-8,
            maxiter=100)

        W = np.reshape(fmin_results, -1)
                
        params_feed_dict = {}
        params_feed_dict[self.W_placeholder] = W        
        self.sess.run(self.set_params_op, feed_dict=params_feed_dict)
        
        if save_checkpoints: self.saver.save(self.sess, self.checkpoint_file, global_step=0)

        if verbose:
            # print('CG training took %s iter.' % model.n_iter_)
            print('After training with CG: ')
            self.print_model_eval()


    def train_with_svm(self, feed_dict, save_checkpoints=True, verbose=True):

        X_train = feed_dict[self.input_placeholder]
        Y_train = feed_dict[self.labels_placeholder]
        num_train_examples = len(Y_train)
        assert len(Y_train.shape) == 1
        assert X_train.shape[0] == Y_train.shape[0]

        print(self.num_train_examples)
        if num_train_examples == self.num_train_examples:
            print('Using normal model')
            model = self.svm_model
        elif num_train_examples == self.num_train_examples - 1:
            print('Using model minus one')
            model = self.svm_model_minus_one
        else:
            #print(self.num_train_examples)
            raise ValueError("feed_dict has incorrect number of training examples")

        model.fit(X_train, Y_train)
        # sklearn returns coefficients in shape num_classes x num_features
        # whereas our weights are defined as num_features x num_classes
        # so we have to tranpose them first.
        if self.use_bias:
            W = np.concatenate((np.reshape(model.coef_.T, -1), model.intercept_), axis=0)
        else:
            W = np.reshape(model.coef_.T, -1)

        self.W = W
        
        params_feed_dict = {}
        params_feed_dict[self.W_placeholder] = W
        self.sess.run(self.set_params_op, feed_dict=params_feed_dict)
        if save_checkpoints: self.saver.save(self.sess, self.checkpoint_file, global_step=0)

        if verbose:
            print('SVM training took %s iter.' % model.n_iter_)
            print('After SVM training: ')
            self.print_model_eval()

        # print('Starting SGD')
        # for step in xrange(100):   
        #     self.sess.run(self.train_op, feed_dict=feed_dict)

        # self.print_model_eval()

    def set_params(self):
        if self.use_bias:
            self.W_placeholder = tf.placeholder(
                tf.float32,
                shape=[self.input_dim + 1],
                name='W_placeholder')
        else:
            self.W_placeholder = tf.placeholder(
                tf.float32,
                shape=[self.input_dim],
                name='W_placeholder')
        set_weights = tf.assign(self.weights, self.W_placeholder, validate_shape=True)
        return [set_weights]
    

    def predictions(self, logits):
        preds = tf.sign(logits, name='preds')
        return preds
 
    #MODIFICATION
    def loss(self, input, logits, labels):
        #logits = self.inference(input)

        pca_inbound = computePCALocation(input,labels,
                                      self.pcas,self.scalers, self.pca_labels,
                                      self.radii, self.elevation)
        self.margin = tf.multiply(
            tf.cast(labels, tf.float32), 
            tf.reshape(logits, [-1]))

        self.margin = tf.multiply(self.margin, pca_inbound)

        indiv_loss_no_reg = smooth_hinge_loss(self.margin, self.temp)
        loss_no_reg = tf.reduce_mean(indiv_loss_no_reg)        
        
        tf.add_to_collection('losses', loss_no_reg)

        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        return total_loss, loss_no_reg, indiv_loss_no_reg


    def adversarial_loss(self, logits, labels):
        wrong_labels = (labels - 1) * -1 # Flips 0s and 1s
        wrong_margins = tf.multiply(
            tf.cast(wrong_labels, tf.float32), 
            tf.reshape(logits, [-1]))  

        indiv_adversarial_loss = -smooth_hinge_loss(wrong_margins, self.temp)
        adversarial_loss = tf.reduce_mean(indiv_adversarial_loss)
        
        return adversarial_loss, indiv_adversarial_loss 


    def get_accuracy_op(self, logits, labels):
        """Evaluate the quality of the logits at predicting the label.
        Args:
          logits: Logits tensor, float - [batch_size, NUM_CLASSES].
          labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).
        Returns:
          A scalar int32 tensor with the number of examples (out of batch_size)
          that were predicted correctly.
        """        
        preds = tf.sign(tf.reshape(logits, [-1]))
        correct = tf.reduce_sum(
            tf.cast(
                tf.equal(
                    preds, 
                    tf.cast(labels, tf.float32)),
                tf.int32))
        return correct / tf.shape(labels)[0]
    def getWeight(self):
        return self.W
