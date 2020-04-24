'''
Created on Apr 22, 2020

Python script that shows the use of a multi-layer perceptron to predict
moment magnitudes from event attributes.

@author: wernerheigl
'''
import os, sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.keras as keras

# this only matters on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

DATA_DIR = '/Users/wernerheigl/ML_data_examples'
DATA_FILE = 'Hauser-MachineLearningSet__Class1and4.csv'
PATH = os.path.join(DATA_DIR, DATA_FILE)

VISUALIZE_DATA = True
VISUALIZE_RESULTS = True
SHOW_TRAINED_VARIABLES = False

# neural network configuration
INPUT_NODES = 3
HIDDEN_NODES = 10
OUTPUT_NODES = 1
EPOCHS = 100
BATCH_SIZE = 50
VALIDATION_SPLIT = 0.3
LEARNING_RATE = 0.0001
MOMENTUM = 0.0
DROPOUT = 0.3


def main():

    print('\n', '# loading data', '\n')
    with open(PATH, 'r') as data_file:
#         cols_to_load = ['Amplitude', 'Magnitude', 'Quality', 'Cloudiness', 'ImageSnRatio']
        cols_to_load = ['Amplitude', 'Magnitude', 'Quality', 'ImageSnRatio']
        df = pd.read_csv(data_file, usecols=cols_to_load)
    print('as read from file:')
    print(df.describe())
    
    print('\n', '# preprocessing data', '\n')
#     df = df.apply(lambda x: np.log(x) if x.name in ['Amplitude', 'Quality', 'Cloudiness', 'ImageSnRatio']
#                   else x)  # makes data more Gaussian
#     df = df.dropna()  # removes Nan and ±inf
    target = df[cols_to_load.pop(1)]  # drop everything except second column
    data = df[cols_to_load]  # cols_to_load now contains remaining columns
    data = data / np.max(data, axis=0)
    assert len(target) == len(data)
    print('\n', 'after normalizing:')
    print(data.describe())
    
    if VISUALIZE_DATA is True:
        data.plot(subplots=True, title='Input Data')
        data.plot(subplots=True, kind='hist', layout=(1, 3),
                title='Input Data Histograms', bins=100)
        pd.plotting.scatter_matrix(data, s=1)
        plt.show()
        sys.exit()

    # set up training and test data
    x_train = data.to_numpy(dtype='float32', copy=True)
    y_train = target.to_numpy(dtype='float32', copy=True)
#     y_train = keras.utils.to_categorical(y_train, OUTPUT_NODES)
    print(y_train)
    
    # Reserve at most 10,000 samples for predictions
    # Note: randint() does drawing w/ replacement
    idx = np.random.randint(0, len(x_train), size=10000)
    x_test = x_train[idx]
    y_test = y_train[idx]
    x_train = np.delete(x_train, idx, axis=0)
    y_train = np.delete(y_train, idx, axis=0)
    
    print('# configuring model', '\n')
    inputs = keras.Input(shape=(INPUT_NODES,), batch_size=BATCH_SIZE)
    hidden = keras.layers.Dense(HIDDEN_NODES,
                                 activation='relu',  # could use 'linear' too
                                 kernel_initializer='glorot_normal')(inputs)
    outputs = keras.layers.Dense(OUTPUT_NODES,
                                 activation='linear')(hidden)
    model = keras.Model(inputs=inputs, outputs=outputs, name='hauser_mlp')
    print(model.summary(), '\n')

    print('# compiling model', '\n')
    optimizerAdam = keras.optimizers.Nadam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizerAdam,
                  loss=keras.losses.mean_absolute_error,
                  metrics=['mean_squared_error'])
    
    print('# training model', '\n')
    history = model.fit(x=x_train, y=y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        shuffle=True,
                        validation_split=VALIDATION_SPLIT,
                        verbose=2)
    print('\n', history.history.keys())
    
    if SHOW_TRAINED_VARIABLES is True:
        print('\n', '# trained variables (weights & bias for each layer & its neurons)', '\n')
        # trainable_variables is an array of type tf.Variable
        for v in model.trainable_variables:
            print(v, '\n')

    print('\n', '# evaluating model on test data', '\n')
    test_loss, test_error = model.evaluate(x_test, y_test, verbose=2)
    
    if VISUALIZE_RESULTS is True:
        plt.figure(figsize=(2 * 6.4, 4.8))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['val_mean_squared_error'])
        plt.plot(history.history['mean_squared_error'])
        plt.hlines(test_error, 0, EPOCHS, colors='red')
        plt.title('model error')
        plt.ylabel('error')
        plt.xlabel('epoch')
        plt.legend(['train', 'val', 'test'], loc='upper right')
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.hlines(test_loss, 0, EPOCHS, colors='red')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val', 'test'], loc='upper right')
        plt.show()

    
if __name__ == '__main__':
    main()
