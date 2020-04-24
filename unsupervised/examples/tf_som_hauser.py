'''
Created on Apr 24, 2020

Python script that shows the use of TensorFlow's self-organizing map
to determine any clustering in the input data.

This script is a lot slower than 'som_cog_hauser.py'.

@author: wernerheigl
'''

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from unsupervised import WTU

# this only matters on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

DATA_DIR = '/Users/wernerheigl/Deep-Learning-with-TensorFlow-2-and-Keras/Chapter 10'
DATA_FILE = 'colors.csv'
PATH = os.path.join(DATA_DIR, DATA_FILE)

VISUALIZE_DATA = True


def normalize(df):
    """
    Normalize input data.
    """
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result.astype(np.float32)


def main():
    
    print('# loading data', '\n')
    with open(PATH, 'r') as data_file:
        df = pd.read_csv(data_file)  # The last column of data file is a label
    data = normalize(df[['R', 'G', 'B']]).values
    name = df['Color-Name'].values
    n_dim = len(df.columns) - 1
    print(df.describe())

    # training data
    colors = data
    color_names = name
    
    som = WTU(30, 30, n_dim, 500, eta=0.1, sigma=10.0)
    som.fit(colors)

    # Get output grid
    image_grid = som.get_centroids()
    
    # Map colors to their closest neurons
    mapped = som.map_vects(colors)
    
    # Plot
    plt.imshow(image_grid)
    plt.title('Color Grid SOM')
    for i, m in enumerate(mapped):
        plt.text(m[1], m[0], color_names[i], ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.5, lw=0))

    idx, loc = som.winner([0.5, 0.5, 0.5])
    print(idx, loc)
    
    
if __name__ == '__main__':
    main()
