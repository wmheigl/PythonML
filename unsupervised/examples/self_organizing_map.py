import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os, sys
from unsupervised.self_organizing_map import *
from numpy.linalg.linalg import norm
import unsupervised.utils as utils

DATA_DIR = '../resources'
DATA_FILE = 'Pioneer_Midkiff_Final_Event_Catalog_Final_2020-02-03.csv'
PATH = os.path.join(DATA_DIR, DATA_FILE)

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

for ar in raw_data.T:
    print(stats.describe(ar))

# utils.plot_data(data=raw_data, labels=labels, show_plot=False)
# utils.plot_data_hist(data=raw_data, labels=labels, show_plot=False)
utils.plot_data_matrix(data=raw_data, labels=labels, show_plot=True)
# sys.exit()

# normalize the data in each column
data = raw_data / raw_data.max(axis=0)
assert data.max(axis=0).all() == 1

# train the SOM
som = SelfOrganizingMap(shape=(10, 10, data.shape[1]),
                        learning_rate=0.01,
                        iterations=200)
bmu_indices = som.learn(data)
bmus = {tuple(i) for i in bmu_indices}
print(f"found {len(bmus)} BMUs")
# print('weights:', som.weights)

# display results & diagnostics
som.plot_history(title='Learning Diagnostics')
som.plot_weights(labels, title='Self-organizing map for PXD-Midkiff events')
plt.show()

