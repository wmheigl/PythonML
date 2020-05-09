'''
Created on May 8, 2020

@author: wernerheigl
'''

import tensorflow as tf
import tensorflow.keras as k

__all__ = ['Encoder', 'Decoder', 'AutoEncoder',
           'SparseEncoder', 'SparseDecoder', 'SparseAutoEncoder',
           'loss', 'train', 'train_loop']
__author__ = 'Werner M. Heigl'


class Encoder(k.layers.Layer):
    '''
    classdocs
    '''

    def __init__(self, hidden_dim):
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


class AutoEncoder(k.Model):
    '''
    classdocs
    '''

    def __init__(self, hidden_dim, original_dim):
        super(AutoEncoder, self).__init__()
        self.loss = []
        self.encoder = Encoder(hidden_dim=hidden_dim)
        self.decoder = Decoder(hidden_dim=hidden_dim, original_dim=original_dim)
    
    def call(self, input_features):
        encoded = self.encoder(input_features)
        reconstructed = self.decoder(encoded)
        return reconstructed


class SparseEncoder(k.layers.Layer):
    '''
    classdocs
    '''

    def __init__(self, hidden_dim):
        super(SparseEncoder, self).__init__()
        self.hidden_layer = k.layers.Dense(units=hidden_dim,
                    activation=tf.nn.relu, activity_regularizer=k.regularizers.l1(10e-5))
    
    def call(self, input_features):
        activation = self.hidden_layer(input_features)
        return activation


class SparseDecoder(k.layers.Layer):
    '''
    classdocs
    '''

    def __init__(self, hidden_dim, original_dim):
        super(SparseDecoder, self).__init__()
        self.output_layer = k.layers.Dense(units=original_dim, activation=tf.nn.relu)
  
    def call(self, encoded):
        activation = self.output_layer(encoded)
        return activation 


class SparseAutoEncoder(k.Model):
    '''
    classdocs
    '''

    def __init__(self, hidden_dim, original_dim):
        super(SparseAutoEncoder, self).__init__()
        self.loss = []
        self.encoder = SparseEncoder(hidden_dim=hidden_dim)
        self.decoder = SparseDecoder(hidden_dim=hidden_dim, original_dim=original_dim)

    def call(self, input_features):
        encoded = self.encoder(input_features)
        reconstructed = self.decoder(encoded)
        return reconstructed


def loss(preds, real):
    return tf.reduce_mean(tf.square(tf.subtract(preds, real)))


def train(loss, model, opt, original):
    with tf.GradientTape() as tape:
        preds = model(original)
        reconstruction_error = loss(preds, original)
        gradients = tape.gradient(reconstruction_error, model.trainable_variables)
        gradient_variables = zip(gradients, model.trainable_variables)
    opt.apply_gradients(gradient_variables)
    return reconstruction_error


def train_loop(model, opt, loss, dataset, epochs=20):
    for epoch in range(epochs):
        epoch_loss = 0
        for step, batch_features in enumerate(dataset):
            loss_values = train(loss, model, opt, batch_features)
            epoch_loss += loss_values
        model.loss.append(epoch_loss)
        print(f'Epoch {epoch + 1}/{epochs}. Loss: {epoch_loss.numpy()}')

