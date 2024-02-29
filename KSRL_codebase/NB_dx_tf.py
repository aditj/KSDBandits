# neural bayesian framework for transition & cost model
import torch
import numpy as np
import os
import time

from datetime import datetime

from scipy.spatial.distance import pdist
from scipy.stats import invgamma
import math
import pickle

import warnings

warnings.filterwarnings("ignore")


class neural_bays_dx_tf(object):
    def __init__(self, args, model, model_type, output_shape, device=None, train_x=None, train_y=None, sigma_n2=0.1,
                 sigma2=0.1):
        self.model = model
        self.model_type = model_type
        self.args = args
        self.device = device
        self.train_x = train_x
        self.train_y = train_y
        #def rewards
        # self.rew = rew
        self.output_shape = output_shape
        self.hidden_dim = 2*model.layers[0].get_input_dim()
        self.beta_s = None
        self.latent_z = None
        self.sigma2 = sigma2  # W prior variance
        self.sigma_n2 = sigma_n2  # noise variacne
        self.eye = np.eye(self.hidden_dim)
        self.mu_w = np.random.normal(loc=0, scale=.01, size=(output_shape, self.hidden_dim))
        self.cov_w = np.array([self.sigma2 * np.eye(self.hidden_dim) for _ in range(output_shape)])

    #primary main code where data is added
    def add_data(self, new_x, new_y):
        if self.train_x is None:
            self.train_x = new_x
            self.train_y = new_y
            #add reward
            # self.rew =  new_r
        else:
            #add the thinning condition : Based on Posterior variane (if posterior variance > threshold)
            self.train_x = np.vstack((self.train_x, new_x))
            self.train_y = np.vstack((self.train_y, new_y))
            # print (torch.is_tensor(self.train_x))
            #add rewards
            # self.rew = np.vstack((self.rew, new_r))
            return self.train_x.shape

    def get_shape(self):
        return self.train_x.shape[0]
    


    def generate_latent_z(self):
        # Update the latent representation of every datapoint collected so far
        new_z = self.get_representation(self.train_x)
        # print ('the shape is' + str(self.train_x.shape))   ## 200 * 4
        self.latent_z = new_z

    #training the latent representation 
    def train(self, epochs = 5):
        self.model.train(self.train_x,self.train_y,epochs=epochs)
        self.generate_latent_z()

    #get the representation
    def get_representation(self, input):
        """
        Returns the latent feature vector from the neural network.
        """
        z = self.model.predict(input, layer = True)
        z = z.squeeze()

        return z


    def check_dim(self):
        print("prior to sampling, check dim as follows: ")
        if self.output_shape == 1:
            print("sampling from cost model")
        else:
            print("sampling from transition model")
        # print("dim of mu: ", np.array(self.mu).shape)
        # print("a = ", self.a)
        # print("dim of a: ", np.array(self.a).shape)
        # print("b = ", self.b)
        # print("cov dim: ", np.array(self.cov).shape)


    def sample(self, parallelize=False):
        d = self.mu_w[0].shape[0]  # hidden_dim
        beta_s = []
        try:
            
            # Here output denotes the dimension of s' or s_t+1.
            # For each output dimension : self.mu_w[i] :  
            #mu_w[i] , cov_w[i] are the mean of the posterior distribution for the bayesian model
            #This samples from the posterior distribution of weights and the samples are saved as beta_s
            #beta_s[i] represents the weight for the first dim

            for i in range(self.output_shape):
                mus = self.mu_w[i]
                covs = self.cov_w[i][np.newaxis, :, :]
                multivariates = np.random.multivariate_normal(mus, covs[0])
                beta_s.append(multivariates)

        except np.linalg.LinAlgError as e:
            # Sampling could fail if covariance is not positive definite
            print('Details: {} | {}.'.format(e.message, e.args))
            multivariates = np.random.multivariate_normal(np.zeros((d)), np.eye(d))
            beta_s.append(multivariates)
        self.beta_s = np.array(beta_s)

    def predict(self, x):
        # Compute last-layer representation for the current context
        z_context = self.get_representation(x)

        # z_context = z_context[np.newaxis, :]
        
        # Apply Thompson Sampling
        vals = (self.beta_s.dot(z_context.T))
        if self.model_type == "dx":
            state = x[:vals.shape[0]] if len(x.shape) == 1 else x[:, :vals.shape[0]]
            return vals.T + state + self.model.layers[len(self.model.layers)-1].biases.eval(session =self.model.sess).squeeze()[:self.output_shape]+ np.random.normal(loc=0, scale=np.sqrt(self.sigma_n2),size = vals.T.shape)
        return vals.T + self.model.layers[len(self.model.layers)-1].biases.eval(session =self.model.sess).squeeze()[:self.output_shape]+np.random.normal(loc=0, scale=np.sqrt(self.sigma_n2),size = vals.T.shape)



    def update_bays_reg(self):

        for i in range(self.output_shape):
            
            # Update  posterior with formulas: \beta | z,y ~ N(mu_q, cov_q)
            z = self.latent_z
            y = self.train_y[:, i] - self.model.layers[len(self.model.layers)-1].biases.eval(session =self.model.sess).squeeze()[i]
            s = np.dot(z.T, z)

            # inv = np.linalg.inv((s/self.sigma_n + 1/self.sigma*self.eye))
            A = s / self.sigma_n2 + 1 / self.sigma2 * self.eye
            B = np.dot(z.T, y) / self.sigma_n2
            reg_coeff = 0

            
            # print ('Shape of A is' + str(A.shape))
            # print ('Shape of B is' + str(B.shape))

            for _ in range(10):
                try:
                    # Compute inv
                    A = A + reg_coeff * self.eye
                    inv = np.linalg.inv(A)
                except Exception as e:
                    # in case computation failed
                    print(e)
                    reg_coeff += 10

                # Store new posterior distributions using inv
                else:
                    self.mu_w[i] = inv.dot(B).squeeze()
                    self.cov_w[i] = inv
                    break
        # print (np.trace(self.cov_w[0]))
        return self.cov_w
        

# currently random thinning
    def compute_posterior_variance(self, new_point):

        #print shape
        new_point = torch.reshape(new_point, (1,-1))

        #get the representation
        z = self.get_representation(new_point)
        z = z.reshape(1,-1)
        
        #compute phi phi trans
        s = np.dot(z.T, z)
        A = s / self.sigma_n2 + 1 / self.sigma2 * self.eye

        #compute inv
        reg_coeff = 0

        for _ in range(10):
            try:
                # Compute inv
                A = A + reg_coeff * self.eye
                inv = np.linalg.inv(A)
            except Exception as e:
                # in case computation failed
                print(e)
                reg_coeff += 10
        
        #compute the post var
        inv = np.linalg.inv(A)         
        post_var = np.trace(inv)
        # print (post_var)
        
        return post_var

    
    #reward based
    def get_high_reward(self):
        self.tr
    

    

    