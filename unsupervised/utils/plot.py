'''
Created on Apr 8, 2020

@author: W. M. Heigl
'''
import matplotlib.pyplot as plt

# exported names at module level
__all__ = ['plot_data', 'plot_data_hist', 'plot_data_matrix']


def plot_data(data=None, labels=None, title='Data', show_plot=False):
    """Plots data.
    
    Arguments:
        data : NumPy ndarray.
            Each row is a feature vector.
        labels : NumPy ndarray
            Feature labels.
        title : String
        show_plot : Bool
            Calls plt.show() if True.
    """
    if data is None:
        print('No data supplied')
        return

    # need some logic to determine number of rows and columns
    n_rows = 1
    n_plots = data.shape[1]
    fig_size = (2 * 6.4, 4.8)
    figure, axes = plt.subplots(nrows=n_rows, ncols=n_plots, figsize=fig_size, constrained_layout=True)
    figure.suptitle(title)
    for index in range(n_plots):
        axes[index].plot(data[:, index])
        if labels is None:
            axes[index].set_title(index)
        else:
            axes[index].set_title(labels[index])          
    if show_plot is True:
        plt.show()


def plot_data_hist(data=None, bins=50, labels=None, title='Data Histograms', show_plot=False):
    """Plots data histograms.
    
    Arguments:
        data : NumPy ndarray.
            Each row is a feature vector.
        bins : Integer
            Number of bins in histograms.
        labels : NumPy ndarray
            Feature labels.
        title : String
        show_plot : Bool
            Calls plt.show() if True.
    """
    if data is None:
        print('No data supplied')
        return

    # need some logic to determine number of rows and columns
    n_rows = 1
    n_plots = data.shape[1]
    fig_size = (2 * 6.4, 4.8)
    figure, axes = plt.subplots(nrows=n_rows, ncols=n_plots, figsize=fig_size, constrained_layout=True)
    figure.suptitle(title)
    for index in range(n_plots):
        axes[index].hist(data[:, index], bins=bins)
        if labels is None:
            axes[index].set_title(index)
        else:
            axes[index].set_title(labels[index])          
    if show_plot is True:
        plt.show()


def plot_data_matrix(data=None, labels=None, title='Data Histograms', show_plot=False):
    """Plots data histograms.
    
    Arguments:
        data : NumPy ndarray.
            Each row is a feature vector.
        bins : Integer
            Number of bins in histograms.
        labels : NumPy ndarray
            Feature labels.
        title : String
        show_plot : Bool
            Calls plt.show() if True.
    """
    if data is None:
        print('No data supplied')
        return
    
    # need some logic to determine number of rows and columns
    n_rows = data.shape[1]
    n_plots = data.shape[1]
    fig_size = (2 * 6.4, 2 * 4.8)
    figure, axes = plt.subplots(nrows=n_rows, ncols=n_plots, figsize=fig_size, constrained_layout=True)
    figure.suptitle(title)
    for row in range(n_plots):
        for col in range(n_plots):
            axes[row, col].scatter(data[:, row], data[:, col], marker='.', s=2)
            axes[row, col].set_xlabel('', visible=False)
            axes[row, col].set_ylabel('', visible=False)
            axes[row, col].set_yticklabels([])
            axes[row, col].set_xticklabels([])
    if labels is None:
        for row in range(n_plots):
            axes[0, row].set_title(row)
            axes[row, 0].set_ylabel(row, visible=True)
    else:
        for row in range(n_plots):
            axes[0, row].set_title(labels[row])
            axes[row, 0].set_ylabel(labels[row], visible=True)
    if show_plot is True:
        plt.show()
