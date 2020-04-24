'''
Created on Apr 24, 2020

Class for self-organizing maps from p. 387 of the book
'Deep Learning With TensorFlow 2 and Keras'.

@author: wernerheigl
'''

__all__ = ['WTU']
__author__ = 'Werner M. Heigl'

import numpy as np
import tensorflow as tf


# Define the Winner Take All units
class WTU(object):
       
    def __init__(self, m, n, dim, num_iterations, eta=0.5, sigma=None):
        """
        m x n : The dimension of the 2D lattice of neurons
        dim : Dimension if the input data
        num_iterations : Total number of training iterations
        eta : Learning rate
        sigma : Radius of neighborhood function
        """
        self._m = m
        self._n = n
        self._neighbourhood = []
        self._topography = []
        self._num_iterations = int(num_iterations) 
        self._learned = False
        self.dim = dim
        
        self.eta = float(eta)
           
        if sigma is None:
            sigma = max(m, n) / 2.0  # Constant radius
        else:
            sigma = float(sigma)
        self.sigma = sigma
            
        print('Network created with dimensions', m, n)
            
        # Weight Matrix and the topography of neurons
        self._W = tf.random.normal([m * n, dim], seed=0)
        self._topography = np.array(list(self._neuron_location(m, n)))
        
    def training(self, x, i):
            m = self._m
            n = self._n 
            
            # Finding the winner and its location
            d = tf.sqrt(tf.reduce_sum(tf.pow(self._W - tf.stack([x for i in range(m * n)]), 2), 1))
            self.WTU_idx = tf.argmin(d, 0)
            
            slice_start = tf.pad(tf.reshape(self.WTU_idx, [1]), np.array([[0, 1]]))
            self.WTU_loc = tf.reshape(tf.slice(self._topography, slice_start, [1, 2]), [2])
            
            # Change learning rate and radius as a function of iterations
            learning_rate = 1 - i / self._num_iterations
            _eta_new = self.eta * learning_rate
            _sigma_new = self.sigma * learning_rate
            
            # Calculating neighborhood function
            distance_square = tf.reduce_sum(tf.pow(tf.subtract(
                self._topography, tf.stack([self.WTU_loc for i in range(m * n)])), 2), 1)
            neighbourhood_func = tf.exp(tf.negative(tf.math.divide(tf.cast(
                distance_square, "float32"), tf.pow(_sigma_new, 2))))
            
            # multiply learning rate with neighborhood function
            eta_into_Gamma = tf.multiply(_eta_new, neighbourhood_func)
            
            # Shape it so that it can be multiplied to calculate dW
            weight_multiplier = tf.stack([tf.tile(tf.slice(
                eta_into_Gamma, np.array([i]), np.array([1])), [self.dim])
                for i in range(m * n)])
            delta_W = tf.multiply(weight_multiplier,
                tf.subtract(tf.stack([x for i in range(m * n)]), self._W))
            new_W = self._W + delta_W
            self._W = new_W
           
    def fit(self, X):
        for i in range(self._num_iterations):
            for x in X:
                self.training(x, i)
        
        # Store a centroid grid for easy retrieval
        centroid_grid = [[] for i in range(self._m)]
        self._Wts = list(self._W)
        self._locations = list(self._topography)
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._Wts[i])
        self._centroid_grid = centroid_grid
        self._learned = True
    
    def winner(self, x):
        idx = self.WTU_idx, self.WTU_loc
        return idx
             
    def _neuron_location(self, m, n):
        """
        Generates a 2D lattice of neurons
        """
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])
                
    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list
        containing the 'n' corresponding centroid locations
        as 1D NumPy arrays.
        """
        if not self._learned:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid

    def map_vects(self, X):
        """
        Maps each input vector to the relevant neuron in the lattice.
        """
        if not self._learned:
            raise ValueError("SOM not trained yet")

        to_return = []
        for vect in X:
            min_index = min([i for i in range(len(self._Wts))],
                            key=lambda x: np.linalg.norm(vect - 
                                                         self._Wts[x]))
            to_return.append(self._locations[min_index])

        return to_return
