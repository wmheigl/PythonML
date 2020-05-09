'''
Created on May 9, 2020

@author: wernerheigl
'''

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
from unsupervised.autoencoder import conv_encoder as enc


def main():
    
    np.random.seed(11)
    tf.random.set_seed(11)
    batch_size = 128
    max_epochs = 10
    filters = [32, 32, 16]

    (x_train, _), (x_test, _) = k.datasets.mnist.load_data()

    x_train = x_train / 255.
    x_test = x_test / 255.
    
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    
    noise = 0.5
    x_train_noisy = x_train + noise * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    
    x_train_noisy = np.clip(x_train_noisy, 0, 1)
    x_test_noisy = np.clip(x_test_noisy, 0, 1)
    
    x_train_noisy = x_train_noisy.astype('float32')
    x_test_noisy = x_test_noisy.astype('float32')
    
    # print(x_test_noisy[1].dtype)

    model = enc.AutoEncoder(filters)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    loss = model.fit(x_train_noisy, x_train,
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
        plt.imshow(tf.reshape(model(x_test_noisy)[index], (28, 28)), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    main()
