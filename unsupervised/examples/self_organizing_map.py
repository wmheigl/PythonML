import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os, sys
from unsupervised.self_organizing_map import *
from numpy.linalg.linalg import norm
import unsupervised.utils as utils

DATA_DIR = '/Users/wernerheigl/ML_data_examples'
DATA_FILE = 'Pioneer_Midkiff_Final_Event_Catalog_Final_2020-02-03.csv'
PATH = os.path.join(DATA_DIR, DATA_FILE)

visualize_data = False
visualize_som = True

print('# loading data', '\n')
with open(PATH, 'r') as data_file:
#     cols_to_load = np.linspace(8, 15, 8, dtype='int')
    cols_to_load = (10, 11, 13, 14, 15)
    labels = np.loadtxt(data_file, delimiter=',', dtype='str', usecols=cols_to_load, max_rows=1)
    raw_data = np.loadtxt(data_file, skiprows=2, delimiter=',', usecols=cols_to_load)
print('data labels =', labels, '\n')
print('data shape: (rows, cols) =', raw_data.shape, '\n')

# transform some of the data
# raw_data[:,0] = np.log(raw_data[:, 0]) # amplitude
# raw_data[:,1] = np.log(raw_data[:,1] + 0.1)    # quality
# raw_data[:,2] += 0.1    # score

print('# descriptive statistics:', 'raw_data', '\n')
for ar in raw_data.T:
    print(stats.describe(ar))

if visualize_data is True:
    utils.plot_data(data=raw_data, labels=labels, show_plot=False)
    utils.plot_data_hist(data=raw_data, labels=labels, show_plot=False)

# normalize the data in each column
raw_data_min = raw_data.min(axis=0)
raw_data_max = raw_data.max(axis=0)
data = (raw_data - raw_data_min) / (raw_data_max - raw_data_min)
assert data.max(axis=0).all() == 1 and data.min(axis=0).all() == 0

print('\n', '# descriptive statistics:', 'normalized raw_data', '\n')
for ar in data.T:
    print(stats.describe(ar))
# print('\n')

if visualize_data is True:
    utils.plot_data_matrix(data=data, labels=labels, show_plot=False)

# SOM
ROWS = 15; COLS = 15
DIM = data.shape[1]  # dimensionality of feature vectors
LEARN_RATE = 0.1  # initial learning rate
ITERATIONS = 18000
som = SelfOrganizingMap(shape=(ROWS, COLS, DIM),
                        learning_rate=LEARN_RATE,
                        iterations=ITERATIONS)
bmu_indices = som.learn(data)
bmus = {tuple(row) for row in bmu_indices}
print(f"found {len(bmus)} BMUs")
# print('weights:', som.weights)

# display results & diagnostics
if visualize_som is True:
    som.plot_history(title='Learning Diagnostics')
    som.plot_weights(labels, title='Self-organizing map for PXD-Midkiff events')
    som.plot_u_matrix()
    plt.show()

