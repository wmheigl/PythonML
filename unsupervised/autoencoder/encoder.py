'''
Created on May 8, 2020

@author: wernerheigl
'''
from unsupervised import autoencoder

__all__ = ['Encoder', 'Decoder', 'Autoencoder']
__author__ = 'Werner M. Heigl'

import tensorflow.keras as k


class Encoder(k.layers.Layer):
    '''
    classdocs
    '''

    def __init__(self, hidden_dim):
        '''
        Constructor
        '''
        super(Encoder, self).__init__()
        self.hidden_layer = k.layers.Dense(units=hidden_dim, activation='relu')
    
    def call(self, input_features):
        activation = self.hidden_layer(input_features)
        return activation


class Decoder(k.layers.Layer):
    '''
    classdocs
    '''
    
    def __init__(self, hidden_dim, original_dim):
        super(Decoder, self).__init__()
        self.output_layer = k.layers.Dense(units=original_dim, activation='relu')
        
    def call(self, encoded):
        activation = self.output_layer(encoded)
        return activation

class Autoencoder(k.Model):
    '''
    classdocs
    '''

    def __init__(self, hidden_dim, original_dim):
        super(Autoencoder, self).__init__()
        self.loss = []
        self.encoder = Encoder(hidden_dim=hidden_dim)
        self.decoder = Decoder(hidden_dim=hidden_dim, original_dim=original_dim)
    
    def call(self, input_features):
        encoded  = self.encoder(input_features)
        reconstructed = self.decoder(encoded)
        return reconstructed
    
        