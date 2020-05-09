'''
Created on May 8, 2020

@author: wernerheigl
'''

import os, sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from unsupervised.autoencoder import encoder as enc


# this only matters on MacOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    
    np.random.seed(11)
    tf.random.set_seed(11)
    batch_size = 256
    max_epochs = 10
    learning_rate = 1e-3
    momentum = 8e-1
    hidden_dim = 128
    original_dim = 784
    
    (x_train, _), (x_test, _) = K.datasets.mnist.load_data()

    x_train = x_train / 255.
    x_test = x_test / 255.
    
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    
    x_train = np.reshape(x_train, (x_train.shape[0], 784))
    x_test = np.reshape(x_test, (x_test.shape[0], 784))
    
    training_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)
    
    model = enc.Autoencoder(hidden_dim=hidden_dim, original_dim=original_dim)
#     opt = tf.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-2)
    enc.train_loop(model, opt, enc.loss, training_dataset, epochs=max_epochs)
    
    plt.plot(range(max_epochs), model.loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    number = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for index in range(number):
        # display original
        ax = plt.subplot(2, number, index + 1)
        plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display reconstruction
        ax = plt.subplot(2, number, index + 1 + number)
        plt.imshow(model(x_test)[index].numpy().reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    main()
