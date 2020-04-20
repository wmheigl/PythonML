#!/usr/bin/env python
# -*- coding: utf-8 -*-

#===============================================================================
# Creates a self-organizing map for the iris data set.
#===============================================================================

import os, sys

import numpy as np
import matplotlib.pyplot as plt

from unsupervised.self_organizing_map import *

DATA_DIR = '/Users/wernerheigl/ML_data_examples'
DATA_FILE = 'iris_data_012.txt'
PATH = os.path.join(DATA_DIR, DATA_FILE)

visualize_som = True

print('# loading data', '\n')
with open(PATH, 'r') as data_file:
    data_x = np.loadtxt(data_file, delimiter=",", usecols=range(0, 4), dtype=np.float64)
    data_y = np.loadtxt(data_file, delimiter=",", usecols=[4], dtype=np.int)
print('data shape: (rows, cols) =', data_x.shape, '\n')

# SOM
ROWS = 30; COLS = 30
DIM = 4  # dimensionality of feature vectors
LEARN_RATE = 0.5  # initial learning rate
ITERATIONS = 5000
som = SelfOrganizingMap(shape=(ROWS, COLS, DIM),
                        learning_rate=LEARN_RATE,
                        iterations=ITERATIONS)
bmu_indices = som.learn(data=data_x, cooling='linear')
bmus = {tuple(row) for row in bmu_indices}
print(f"found {len(bmus)} BMUs")

# display results & diagnostics
if visualize_som is True:
    som.plot_history(figsize=(6.4, 2.4), title='Learning Diagnostics')
    title='Self-organizing map for COG-Hauser events'
    som.plot_components(figsize=(6.4, 2.4), title=title)
    som.plot_u_matrix()
    plt.show()

