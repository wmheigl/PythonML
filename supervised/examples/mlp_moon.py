# -*- coding: utf-8 -*-
"""Multi-layer perceptron to classify microseismic events."""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress INFO and WARNING messages

DATA_DIR = '/Users/wernerheigl/dl4j-examples-data/dl4j-examples/classification'
DATA_TRAIN = 'moon_data_train.csv'
DATA_TEST = 'moon_data_eval.csv'
PATH_TRAIN = os.path.join(DATA_DIR, DATA_TRAIN)
PATH_TEST = os.path.join(DATA_DIR, DATA_TEST)

INPUT_NODES = 2
HIDDEN_NODES = 10
OUTPUT_NODES = 2
EPOCHS = 50
BATCH_SIZE = 50
VALIDATION_SPLIT = 0.2
LEARNING_RATE=0.01
MOMENTUM=0.0
DROPOUT=0.3

# loading the 'moon' dataset
print('# loading data', '\n')
data_train = np.loadtxt(PATH_TRAIN, delimiter=',')
x_train = data_train[:, 1:].astype('float32')  # (x,y) coordinates
y_train = data_train[:, 0].astype('float32')   # labels
print('training data shape:', 'x', x_train.shape, 'y', y_train.shape)
print('1st entry:', 'x=', x_train[0, :], 'y=', y_train[0], '\n')

data_test = np.genfromtxt(PATH_TEST, delimiter=',')
x_test = data_test[:, 1:].astype('float32')  # (x,y) coordinates
y_test = data_test[:, 0].astype('float32')  # labels
print('test data shape:', 'x=', x_test.shape, 'y=', y_test.shape)
print('1st entry:', 'x=', x_test[0, :], 'y=', y_test[0], '\n')

# one-hot representation of labels
y_train = keras.utils.to_categorical(y_train, OUTPUT_NODES)
y_test = keras.utils.to_categorical(y_test, OUTPUT_NODES)

print('# configuring model', '\n')
inputs = keras.Input(shape=(INPUT_NODES,), batch_size=BATCH_SIZE)
hidden1 = keras.layers.Dense(HIDDEN_NODES,
                             activation='relu',
                             kernel_initializer='glorot_normal')
x = hidden1(inputs)
outputs = keras.layers.Dense(OUTPUT_NODES,
                             activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs, name='moon_mlp')
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

print('\n', '# trained variables', '\n')
# trainable_variables is an array of type tf.Variable
for v in model.trainable_variables:
    print(v, '\n')

print('\n', '# evaluating model on test data', '\n')
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print(history.history.keys())

sys.exit()

plt.plot(history.history['val_accuracy'])
# plt.figure(figsize=(2 * 6.4, 4.8))
# plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.show()
plt.hlines(test_acc, 0, EPOCHS, colors='red')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val', 'test'], loc='lower right')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.hlines(test_loss, 0, EPOCHS, colors='red')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val', 'test'], loc='upper right')
