'''
Created on May 9, 2020

@author: wernerheigl
'''

import tensorflow.keras as k
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

__all__ = ['Encoder', 'Decoder', 'AutoEncoder']
__author__ = 'Werner M. Heigl'


class Encoder(k.layers.Layer):
    '''
    classdocs
    '''

    def __init__(self, filters):
        super(Encoder, self).__init__()
        self.conv1 = Conv2D(filters=filters[0], kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv2 = Conv2D(filters=filters[1], kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv3 = Conv2D(filters=filters[2], kernel_size=3, strides=1, activation='relu', padding='same')
        self.pool = MaxPooling2D((2, 2), padding='same')
    
    def call(self, input_features):
        x = self.conv1(input_features)
        # print("Ex1", x.shape)
        x = self.pool(x)
        # print("Ex2", x.shape)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        return x


class Decoder(k.layers.Layer):
    '''
    classdocs
    '''

    def __init__(self, filters):
        super(Decoder, self).__init__()
        self.conv1 = Conv2D(filters=filters[2], kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv2 = Conv2D(filters=filters[1], kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv3 = Conv2D(filters=filters[0], kernel_size=3, strides=1, activation='relu', padding='valid')
        self.conv4 = Conv2D(1, 3, 1, activation='sigmoid', padding='same')
        self.upsample = UpSampling2D((2, 2))
  
    def call(self, encoded):
        x = self.conv1(encoded)
        # print("dx1", x.shape)
        x = self.upsample(x)
        # print("dx2", x.shape)
        x = self.conv2(x)
        x = self.upsample(x)
        x = self.conv3(x)
        x = self.upsample(x)
        return self.conv4(x)

    
class Autoencoder(k.Model):
    '''
    classdocs
    '''

    def __init__(self, filters):
        super(Autoencoder, self).__init__()
        self.loss = []
        self.encoder = Encoder(filters)
        self.decoder = Decoder(filters)

    def call(self, input_features):
        # print(input_features.shape)
        encoded = self.encoder(input_features)
        # print(encoded.shape)
        reconstructed = self.decoder(encoded)
        # print(reconstructed.shape)
        return reconstructed
