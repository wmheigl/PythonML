'''
Created on Apr 20, 2020

@author: wernerheigl
'''
import os, sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    
    DATA_DIR = '/Users/wernerheigl/ML_data_examples'
    # DATA_FILE = 'COG_Hauser_no_outliers.csv'
    DATA_FILE = 'Hauser-MachineLearningSet__Class1and4.csv'
    PATH = os.path.join(DATA_DIR, DATA_FILE)
    
    VISUALIZE_DATA = False
    VISUALIZE_RESULTS = True
    SHOW_TRAINED_VARIABLES = False

    # neural network configuration
    INPUT_NODES = 4
    HIDDEN_NODES = 10
    OUTPUT_NODES = 2
    EPOCHS = 5
    BATCH_SIZE = 50
    VALIDATION_SPLIT = 0.3
    LEARNING_RATE = 0.001
    MOMENTUM = 0.0
    DROPOUT = 0.3
    
    print('# loading data', '\n')
    with open(PATH, 'r') as data_file:
        cols_to_load = ['Amplitude', 'Quality', 'Cloudiness', 'ImageSnRatio', 'Class']
        df = pd.read_csv(data_file, usecols=cols_to_load)
    print('as read from file:')
    print(df.describe())
    
    df = df.apply(np.log)  # makes data more Gaussian
    df = df.dropna()  # removes Nan and ±inf
    data = df[cols_to_load[:-1]]  # drop last column
    labels = df[cols_to_load.pop()]  # drop everything except last column
    print('\nafter transformation and cleaning:')
    print(data.describe())
    
    if VISUALIZE_DATA is True:
        data.plot(subplots=True, title='Input Data')
        data.plot(subplots=True, kind='hist', layout=(1, 4),
                title='Input Data Histograms', bins=100)
        pd.plotting.scatter_matrix(data, s=1)
        plt.show()

    # set up training and test data
    x_train = data.to_numpy(dtype='float32', copy=True)
    y_train = labels.to_numpy(dtype='float32', copy=True)
    y_train = keras.utils.to_categorical(y_train, OUTPUT_NODES)
    
    # Reserve at most 10,000 samples for predictions
    # Note: randint() is drawing w/ replacement
    idx = np.random.randint(0, len(x_train), size=10000)
    x_test = x_train[idx]
    y_test = y_train[idx]
    x_train = np.delete(x_train, idx, axis=0)
    y_train = np.delete(y_train, idx, axis=0)
    
    print('# configuring model', '\n')
    inputs = keras.Input(shape=(INPUT_NODES,), batch_size=BATCH_SIZE)
    hidden1 = keras.layers.Dense(HIDDEN_NODES,
                                 activation='relu',
                                 kernel_initializer='glorot_normal')(inputs)
    outputs = keras.layers.Dense(OUTPUT_NODES,
                                 activation='softmax')(hidden1)
    model = keras.Model(inputs=inputs, outputs=outputs, name='hauser_mlp')
    print(model.summary(), '\n')

    print('# compiling model', '\n')
    optimizerSGD = keras.optimizers.SGD(learning_rate=LEARNING_RATE,
                                        momentum=MOMENTUM,
                                        nesterov=True)
    optimizerRMSprop = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE,
                                                rho=0.9,
                                                momentum=MOMENTUM)
    optimizerAdam = keras.optimizers.Nadam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizerRMSprop,
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    print('# training model', '\n')
    history = model.fit(x=x_train, y=y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        shuffle=True,
                        validation_split=VALIDATION_SPLIT,
                        verbose=1)

    if SHOW_TRAINED_VARIABLES is True:
        print('\n', '# trained variables (weights & bias for each neuron)', '\n')
        # trainable_variables is an array of type tf.Variable
        for v in model.trainable_variables:
            print(v, '\n')

    print('\n', '# evaluating model on test data', '\n')
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    # confusion matrix for test data
    print('\n', '# confusion matrix for test data', '\n')
    labels = np.argmax(y_test, axis=1)
    predictions = np.argmax(model.predict(x_test), axis=1)
    m = tf.math.confusion_matrix(labels, predictions, num_classes=2)
    tf.print(m, summarize=-1)

    if VISUALIZE_RESULTS is True:
        plt.figure(figsize=(2 * 6.4, 4.8))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['val_accuracy'])
        plt.plot(history.history['accuracy'])
        plt.hlines(test_acc, 0, EPOCHS, colors='red')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.hlines(test_loss, 0, EPOCHS, colors='red')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()


if __name__ == '__main__':
    main()
