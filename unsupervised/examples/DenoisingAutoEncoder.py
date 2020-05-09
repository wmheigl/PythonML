'''
Created on May 9, 2020

@author: wernerheigl
'''

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
from unsupervised.autoencoder import encoder as enc

# this only matters on MacOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    
    np.random.seed(11)
    tf.random.set_seed(11)
    batch_size = 256
    max_epochs = 20
    learning_rate = 1e-3
    momentum = 8e-1
#     hidden_dim = 784 + 392
    hidden_dim = 128
    original_dim = 784
    
    (x_train, _), (x_test, _) = k.datasets.mnist.load_data()

    x_train = x_train / 255.
    x_test = x_test / 255.
    
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    
    x_train = np.reshape(x_train, (x_train.shape[0], 784))
    x_test = np.reshape(x_test, (x_test.shape[0], 784))
    
    # Generate corrupted MNIST images by adding noise with normal dist
    # centered at 0.5 and std=0.5
    noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
    x_train_noisy = x_train + noise
    noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
    x_test_noisy = x_test + noise

    model = enc.Autoencoder(hidden_dim=hidden_dim, original_dim=original_dim)
    model.compile(loss='mse', optimizer='adam')
    loss = model.fit(x_train_noisy,
                    x_train,
                    validation_data=(x_test_noisy, x_test),
                    epochs=max_epochs,
                    batch_size=batch_size)

    plt.plot(range(max_epochs), loss.history['loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    number = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for index in range(number):
        # display original
        ax = plt.subplot(2, number, index + 1)
        plt.imshow(x_test_noisy[index].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(2, number, index + 1 + number)
        plt.imshow(model(x_test_noisy)[index].numpy().reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    main()
