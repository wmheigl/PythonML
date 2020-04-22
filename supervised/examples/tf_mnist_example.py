'''
Created on Apr 21, 2020

@author: wernerheigl
'''
import sys

from tensorflow import keras

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def main():
    
    print('\n', '# building model', '\n')
    inputs = keras.Input(shape=(784,), name='digits')
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(10, activation='linear', name='predictions')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    print('\n', '# loading data', '\n')
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the data (these are Numpy arrays)
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255
    
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    
    # Reserve 10,000 samples for validation
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]
    
    model.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer
              # Loss function to minimize
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # List of metrics to monitor
              metrics=['sparse_categorical_accuracy'])
    
    print('# Fit model on training data')
    history = model.fit(x_train, y_train,
                        batch_size=64,
                        epochs=1,
                        # We pass some validation for
                        # monitoring validation loss and metrics
                        # at the end of each epoch
                        validation_data=(x_val, y_val))
    
#     print('\nhistory dict:', history.history)
    
    # Evaluate the model on the test data using `evaluate`
    print('\n', '# Evaluate on test data')
    results = model.evaluate(x_test, y_test, batch_size=128, verbose=0)  # verbose=1 clears Eclipse console
    print('test loss, test acc:', results)
    
    # confusion matrix for test data
    print('\n', '# confusion matrix for test data', '\n')
    predictions = np.argmax(model.predict(x_test), axis=1)
    print(predictions[:10])
    m = tf.math.confusion_matrix(y_test, predictions)
    tf.print(m, summarize=-1)


if __name__ == '__main__':
    main()
