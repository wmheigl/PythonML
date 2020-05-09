#!/usr/bin/env python
# -*- coding: utf-8 -*-

#===============================================================================
# Demonstrates data loading and plotting with pandas.
#===============================================================================

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = '/Users/wernerheigl/ML_data_examples'
# DATA_FILE = 'COG_Hauser_no_outliers.csv'
DATA_FILE = 'Hauser-MachineLearningSet__Class1and4.csv'
PATH = os.path.join(DATA_DIR, DATA_FILE)

VISUALIZE_DATA = True

print('# loading data', '\n')
with open(PATH, 'r') as data_file:
    cols_to_load = ['Amplitude', 'Quality', 'Cloudiness', 'ImageSnRatio']
    df = pd.read_csv(data_file, usecols=cols_to_load)
print(df.describe())

# transform some of the data (makes data more Gaussian-looking)
df = df[cols_to_load].apply(np.log)
df = df[np.isfinite(df) | np.isnan(df)]
print(df.describe())

if VISUALIZE_DATA is True:
    df.plot(subplots=True, title='Input Data')
    df.plot(subplots=True, kind='hist', layout=(1, 4),
            title='Input Data Histograms', bins=100)
    pd.plotting.scatter_matrix(df, s=1)
plt.show()
