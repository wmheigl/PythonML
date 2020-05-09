'''
Created on May 9, 2020

@author: wernerheigl
'''

import tensorflow as tf
import numpy as np


class RestrictedBoltzmannMachine(object):
    
    def __init__(self, input_size, output_size, lr=1.0, batchsize=100):
        """
        m: Number of neurons in visible layer
        n: number of neurons in hidden layer
        """
        # Defining the hyperparameters
        self._input_size = input_size  # Size of Visible
        self._output_size = output_size  # Size of outp
        self.learning_rate = lr  # The step used in gradient descent
        self.batchsize = batchsize  # The size of how much data will be used for training per sub iteration
        
        # Initializing weights and biases as matrices full of zeroes
        self.w = tf.zeros([input_size, output_size], np.float32)  # Creates and initializes the weights with 0
        self.hb = tf.zeros([output_size], np.float32)  # Creates and initializes the hidden biases with 0
        self.vb = tf.zeros([input_size], np.float32)  # Creates and initializes the visible biases with 0

    # Forward Pass
    def prob_h_given_v(self, visible, w, hb):
        # Sigmoid 
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

    # Backward Pass
    def prob_v_given_h(self, hidden, w, vb):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)
    
    # Generate the sample probability
    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs))))

    # Training method for the model
    def train(self, X, epochs=10):
               
        loss = []
        for epoch in range(epochs):
            # For each step/batch
            for start, end in zip(range(0, len(X), self.batchsize), range(self.batchsize, len(X), self.batchsize)):
                batch = X[start:end]
                    
                # Initialize with sample probabilities
                    
                h0 = self.sample_prob(self.prob_h_given_v(batch, self.w, self.hb))
                v1 = self.sample_prob(self.prob_v_given_h(h0, self.w, self.vb))
                h1 = self.prob_h_given_v(v1, self.w, self.hb)
                    
                # Create the Gradients
                positive_grad = tf.matmul(tf.transpose(batch), h0)
                negative_grad = tf.matmul(tf.transpose(v1), h1)
                    
                # Update learning rates 
                self.w = self.w + self.learning_rate * (positive_grad - negative_grad) / tf.dtypes.cast(tf.shape(batch)[0], tf.float32)
                self.vb = self.vb + self.learning_rate * tf.reduce_mean(batch - v1, 0)
                self.hb = self.hb + self.learning_rate * tf.reduce_mean(h0 - h1, 0)
                    
            # Find the error rate
            err = tf.reduce_mean(tf.square(batch - v1))
            print (f'Epoch: {epoch} reconstruction error: {err}')
            loss.append(err)
                    
        return loss
        
    # Create expected output for our DBN
    def rbm_output(self, X):
        out = tf.nn.sigmoid(tf.matmul(X, self.w) + self.hb)
        return out
    
    def rbm_reconstruct(self, X):
        h = tf.nn.sigmoid(tf.matmul(X, self.w) + self.hb)
        reconstruct = tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.w)) + self.vb)
        return reconstruct
        
