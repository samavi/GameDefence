import tensorflow as tf
from scipy import fmin_ncg
from influence.hessians import hessian_vector_product
class SoftMaxModel:
    
    def __init__(self,X_train,Y_train,X_validation, Y_validation, learning_rate=0.01):

        #training data
        self.numFeatures = X_train.shape[1]
        self.numClass = Y_train.shape[1] #requires one hot encoding
        self.num_train_data = X_train.shape[0]

        self.X_train=X_train
        self.Y_train=Y_train
        self.X_validation=X_validation
        self.Y_validation = Y_validation

        #placeholders
        self.xhold = tf.placeholder(tf.float32, [None, self.numFeatures])
        self.yhold = tf.placeholder(tf.float32, [None, self.numClass])

        #model
        self.W = tf.Variable(tf.zeros([self.numFeatures, self.numClass]))
        self.b = tf.Variable(tf.zeros([self.numClass]))
        self.learning_rate=learning_rate
        
        #prediction algorithm
        apply_weights_OP = tf.matmul(self.xhold, self.W, name="apply_weights")
        add_bias_OP = tf.add(apply_weights_OP, self.b, name="add_bias") 
        activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")
        
        self.pred_model = activation_OP

        #cost
        self.cost = tf.reduce_mean(tf.squared_difference(self.yhold, self.pred_model))
        #training operation
        self.training_OP = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
        #accuracy check operation
        correct_prediction = tf.equal(tf.argmax(self.pred_model,axis=1), tf.argmax(self.yhold,axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #session
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        #for gradient computation
        self.oldW = tf.Variable(self.W)
        self.oldB = tf.Variable(self.b)
        
        self.copyW = self.oldW.assign(self.W)
        self.copyB = self.oldB.assign(self.b)

        self.wlgrad= tf.gradients(xs = [self.W,self.b], ys = self.cost)

        self.v_placeholder =tf.placeholder(tf.float32, shape=W.get_shape())
        self.hessian_vector = hessian_vector_product(self.cost, self.W, self.v_placeholder)

        #misc
        self.vec_to_list = self.get_vec_to_list_fn()

    def train_model(self,training_epochs=3000, retrain = False):
        # Initialize the variables (i.e. assign their default value)
        
        display_step = 1000
        if retrain == True:
            #re-initialize all variables
            init = tf.global_variables_initializer()

        #store the original W and B
        self.sess.run(self.copyW)
        self.sess.run(self.copyB)
        
        # Training cycle
        for epoch in range(training_epochs):
           
            step = self.sess.run(self.training_OP, feed_dict={self.xhold: self.X_train,
                                                           self.yhold: self.Y_train})
            a,c =self.sess.run([self.accuracy,self.cost], feed_dict= {self.xhold: self.X_train
                                                                 , self.yhold: self.Y_train})
            
            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),
                       "accuracy=", "{:.9f}".format(a))
                    

        print ("Optimization Finished!")
            # Test model
    def test_model(self):
        print ("Accuracy:", self.sess.run(self.accuracy,feed_dict ={self.xhold: self.X_validation,
                                                self.yhold: self.Y_validation}))

    def get_gradient_dw_dx(self, trained = True):
        #do (new dw - old dw)/change in x
        feed_dict ={self.xhold: self.X_validation,
                                                self.yhold: self.Y_validation}
        print(self.sess.run(self.wlgrad,feed_dict = feed_dict))

############TODO##########
    def get_grad_of_influence_wrt_input(self, train_indices, test_indices, 
        approx_type='cg', approx_params=None, force_refresh=True, verbose=True, test_description=None,
        loss_type='normal_loss'):
        """
        If the loss goes up when you remove a point, then it was a helpful point.
        So positive influence = helpful.
        If we move in the direction of the gradient, we make the influence even more positive, 
        so even more helpful.
        Thus if we want to make the test point more wrong, we have to move in the opposite direction.
        """

        # Calculate v_placeholder (gradient of loss at test point)
        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices, loss_type=loss_type)            

        if verbose: print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))
        
        start_time = time.time()

        if test_description is None:
            test_description = test_indices

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' % (self.model_name, approx_type, loss_type, test_description))
        
        if os.path.exists(approx_filename) and force_refresh == False:
            inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
            if verbose: print('Loaded inverse HVP from %s' % approx_filename)
        else:            
            inverse_hvp = self.get_inverse_hvp(
                test_grad_loss_no_reg_val,
                approx_type,
                approx_params,
                verbose=verbose)
            np.savez(approx_filename, inverse_hvp=inverse_hvp)
            if verbose: print('Saved inverse HVP to %s' % approx_filename)            
        
        duration = time.time() - start_time
        if verbose: print('Inverse HVP took %s sec' % duration)

        grad_influence_wrt_input_val = None

        for counter, train_idx in enumerate(train_indices):
            # Put in the train example in the feed dict
            grad_influence_feed_dict = self.fill_feed_dict_with_one_ex(
                self.data_sets.train,  
                train_idx)

            self.update_feed_dict_with_v_placeholder(grad_influence_feed_dict, inverse_hvp)

            # Run the grad op with the feed dict
            current_grad_influence_wrt_input_val = self.sess.run(self.grad_influence_wrt_input_op, feed_dict=grad_influence_feed_dict)[0][0, :]            
            
            if grad_influence_wrt_input_val is None:
                grad_influence_wrt_input_val = np.zeros([len(train_indices), len(current_grad_influence_wrt_input_val)])

            grad_influence_wrt_input_val[counter, :] = current_grad_influence_wrt_input_val

        return grad_influence_wrt_input_val
    
    def get_inverse_hvp_cg(self, v, verbose):
        fmin_loss_fn = self.get_fmin_loss_fn(v)
        fmin_grad_fn = self.get_fmin_grad_fn(v)
        cg_callback = self.get_cg_callback(v, verbose)

        fmin_results = fmin_ncg(
            f=fmin_loss_fn,
            x0=np.concatenate(v),
            fprime=fmin_grad_fn,
            fhess_p=self.get_fmin_hvp,
            callback=cg_callback,
            avextol=1e-8,
            maxiter=100)
    def get_fmin_loss_fn(self, v):

        def get_fmin_loss(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))

            return 0.5 * np.dot(np.concatenate(hessian_vector_val), x) - np.dot(np.concatenate(v), x)
        return get_fmin_loss

        return self.vec_to_list(fmin_results)
    
    def get_fmin_grad_fn(self, v):
        def get_fmin_grad(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))
            
            return np.concatenate(hessian_vector_val) - np.concatenate(v)
        return get_fmin_grad
    
    def get_cg_callback(self, v, verbose):
        fmin_loss_fn = self.get_fmin_loss_fn(v)
        
        def fmin_loss_split(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))

            return 0.5 * np.dot(np.concatenate(hessian_vector_val), x), -np.dot(np.concatenate(v), x)

        def cg_callback(x):
            # x is current params
            v = self.vec_to_list(x)
            idx_to_remove = 5

            single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.X_train,
                                                                     self.Y_train, idx_to_remove)      
            train_grad_loss_val = self.sess.run(self.wlgrad, feed_dict=single_train_feed_dict)
            predicted_loss_diff = np.dot(np.concatenate(v), np.concatenate(train_grad_loss_val)) / self.num_train_data

            if verbose:
                print('Function value: %s' % fmin_loss_fn(x))
                quad, lin = fmin_loss_split(x)
                print('Split function value: %s, %s' % (quad, lin))
                print('Predicted loss diff on train_idx %s: %s' % (idx_to_remove, predicted_loss_diff))

        return cg_callback
    
    def get_fmin_hvp(self, x, p):
        hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(p))

        return np.concatenate(hessian_vector_val)

    def minibatch_hessian_vector_val(self, v):

        num_examples = self.num_train_data
        batch_size = self.num_train_data

        num_iter = int(num_examples / batch_size)

        hessian_vector_val = None
        for i in range(num_iter):
            feed_dict = {self.xhold = self.X_train, self.yhold = self.Y_train}

            hessian_vector_val_temp = self.sess.run(self.hessian_vector, feed_dict=feed_dict)
            if hessian_vector_val is None:
                hessian_vector_val = [b / float(num_iter) for b in hessian_vector_val_temp]
            else:
                hessian_vector_val = [a + (b / float(num_iter)) for (a,b) in zip(hessian_vector_val, hessian_vector_val_temp)]
            
        hessian_vector_val = [a + 0 * b for (a,b) in zip(hessian_vector_val, v)]

        return hessian_vector_val
    
    def get_test_grad_loss_no_reg_val(self, test_indices, batch_size=100, loss_type='normal_loss'):

        if loss_type == 'normal_loss':
            op = self.wlgrad
        #elif loss_type == 'adversarial_loss':
         #   op = self.grad_adversarial_loss_op
        else:
            raise ValueError, 'Loss must be specified'

        if test_indices is not None:
            num_iter = int(np.ceil(len(test_indices) / batch_size))

            test_grad_loss_no_reg_val = None
            for i in range(num_iter):
                start = i * batch_size
                end = int(min((i+1) * batch_size, len(test_indices)))

                test_feed_dict = self.fill_feed_dict_with_some_ex(self.data_sets.test, test_indices[start:end])

                temp = self.sess.run(op, feed_dict=test_feed_dict)

                if test_grad_loss_no_reg_val is None:
                    test_grad_loss_no_reg_val = [a * (end-start) for a in temp]
                else:
                    test_grad_loss_no_reg_val = [a + b * (end-start) for (a, b) in zip(test_grad_loss_no_reg_val, temp)]

            test_grad_loss_no_reg_val = [a/len(test_indices) for a in test_grad_loss_no_reg_val]

        else:
            test_grad_loss_no_reg_val = self.minibatch_mean_eval([op], self.data_sets.test)[0]
        
        return test_grad_loss_no_reg_val

    ###########MISC#######
    def get_vec_to_list_fn(self):
        params_val = self.sess.run(self.W)
        #self.num_params = len(np.concatenate(params_val))        
        #print('Total number of parameters: %s' % self.num_params)


        def vec_to_list(v):
            return_list = []
            cur_pos = 0
            for p in params_val:
                return_list.append(v[cur_pos : cur_pos+len(p)])
                cur_pos += len(p)

            assert cur_pos == len(v)
            return return_list

        return vec_to_list


    def fill_feed_dict_with_one_ex(self, X,Y, target_idx):
        input_feed = X[target_idx, :].reshape(1, -1)
        labels_feed = Y[target_idx].reshape(-1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict
    
        
