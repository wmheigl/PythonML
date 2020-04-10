'''
Created on Apr 2, 2020

@author: W. M. Heigl
'''
import numpy as np
import matplotlib.pyplot as plt
import sys

# exported names at module level
__all__ = ['SelfOrganizingMap']


class SelfOrganizingMap(object):
    """A self-organizing map.
    
    A self-organizing map (SOM) is a type of neural network in which the neurons
    are conceptually arranged in a 2D rectangular grid. In contrast with
    layer-based neural networks the neurons in a SOM are not connected to each
    other, only to the input.
    """

    def __init__(self, shape=(0, 0, 0), learning_rate=0.01, iterations=200):
        """Initializes a self-organizing map.
        
        The number of neurons in the network is shape[0]*shape[1]. Every node
        has shape[2] weights (equal to the number of features).
        
        Arguments:
        ----------
            shape : Tuple
                The size of the network is (rows, cols, weights)
            learning_rate : Float
                The learning rate of the network.
            iterations : Integer
                The number of iterations to train the network.
        """
        self.__shape = shape
        self.__learning_rate = learning_rate
        self.__iterations = iterations
        # the 'state' of an instance during learning
        self.__weights = np.random.random(shape)
        # various measures of progress during learning
        self.__history = {'radius' : [], 'learn_rate' : [], 'bmu_distance' : []}

    @property
    def history(self):
        return self.__history

    @property
    def iterations(self):
        return self.__iterations

    @property
    def learning_rate(self):
        return self.__learning_rate
    
    @property
    def shape(self):
        return self.__shape
    
    @property
    def weights(self):
        return self.__weights

    def learn(self, data):
        """Trains the SOM using the provided data.
        
        Arguments:
            data : NumPy ndarray
                The data as a 2D NumPy array consisting of rows of feature vectors.
            
        Returns
        ----------
        bmu_idx_list : List
            A list of BMU indices
        distance_list : List
            A list of squared distances
        radius_list : List
        
        learn_rate_list : List
            The data as a 2D NumPy array consisting of rows of feature vectors.
        """
        if self.__shape == (0, 0, 0) or data == None:
            print(f"shape={self.__shape} -- nothing to be done")
            return
        
        bmu_idx_list = []
        
        # this is to save real estate
        m = data.shape[0]   # no. of feature vectors
        n = data.shape[1]   # no. of features
        
        initial_radius = max(self.__shape[0], self.__shape[1]) / 2
        learning_rate = self.__learning_rate
        time_constant = self.__iterations / np.log(initial_radius)

        self.__history['radius'].append(initial_radius)
        self.__history['learn_rate'].append(learning_rate)
        
        for i in range(self.__iterations):
            if(i % 100 == 0):
                print('iteration ', i)
            
            # select a training example at random
            t = data[np.random.randint(0, m)]
#             t = data[i]

            # find its Best Matching Unit
            bmu, bmu_idx, dist = self.__find_bmu(t)
            
            bmu_idx_list.append(bmu_idx)
            
            # update the learning parameters
            radius = initial_radius * np.exp(-i / time_constant)
            radius_squared = radius ** 2
            learning_rate = learning_rate * np.exp(-i / (2000 * self.__iterations))
            
            self.__history['radius'].append(radius)
            self.__history['learn_rate'].append(learning_rate)
            self.__history['bmu_distance'].append(dist)
            
            # now we know the BMU, update its weight vector to move closer to input
            # and move its neighbors closer too
            # by a factor proportional to their 2-D distance from the BMU
            for x in range(self.__weights.shape[0]):
                for y in range(self.__weights.shape[1]):
                    w = self.__weights[x, y, :]
                    # get the squared Euclidean distance
                    w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
                    # if the distance is within the current neighbourhood radius
                    if w_dist <= radius_squared:
                        # calculate the degree of neighborhood (based on the 2-D distance)
#                         neighborhood = self.__calculate_influence(w_dist, radius)
                        neighborhood = np.exp(-w_dist / (2 * radius_squared))
                        # now update the neuron's weight using the formula:
                        # new w = old w + (learning rate * neighborhood * delta)
                        # where delta = input vector (t) - old w
                        new_w = w + (learning_rate * neighborhood * (t - w))
                        # commit the new weight
                        self.__weights[x, y, :] = new_w
                        
        return bmu_idx_list

    def __find_bmu(self, t):
        """Finds the best matching unit BMU in the SOM for the feature vector t.
        
        Parameters
        ------------
        t : ndarray
            The feature vector.
            
        Returns
        ---------
        (bmu, bmu_idx, min_dist) : tuple
            A tuple where bmu contains the weights of the BMU, bmu_idx its
            location in the SOM, and min_dist between the input vector and
            the weights of the BMU
        """
        bmu_idx = np.array([0, 0])
        # set the initial minimum distance to a huge number
        min_dist = np.iinfo(np.int).max
        # calculate the high-dimensional distance between each neuron and the input
        for x in range(self.__weights.shape[0]):
            for y in range(self.__weights.shape[1]):
                w = self.__weights[x, y, :]
                # don't bother with actual Euclidean distance, to avoid expensive sqrt operation
                square_dist = np.sum((w - t) ** 2)
#                 square_dist = np.sum(np.abs(w - t)) # L1 norm
                if square_dist < min_dist:
                    min_dist = square_dist
#                     bmu_idx = np.array([x, y])
                    np.put(bmu_idx, [0, 1], [x, y])
        # get vector corresponding to bmu_idx
        bmu = self.__weights[bmu_idx[0], bmu_idx[1], :]
        # return the (bmu, bmu_idx) tuple
        return (bmu, bmu_idx, min_dist)

    def plot_history(self, title='Learning History', show_plot=False):
        """Plots learning history
    
        Arguments:
            title : String
                Title of the plot.
            show_plot : Bool
                Calls plt.show() if True.
        """
        n_plots = len(self.__history)
        figure, axis = plt.subplots(nrows=1, ncols=n_plots, figsize=(2 * 6.4, 4.8), constrained_layout=True)
        figure.suptitle(title)
        index = 0
        for key in self.__history:
            axis[index].plot(self.__history[key])
            axis[index].set_title(key)
#             axis[index].get_xaxis().set_visible(False)
#             axis[index].get_yaxis().set_visible(False)
            index += 1
        if show_plot is True:
            plt.show()
        
    def plot_weights(self, labels=None, title='SOM Model Components', show_plot=False):
        """Plots SOM weights (model components) as images.
    
        Arguments:
            labels : NumPy ndarray
                Feature labels.
            title : String
                Title of the plot.
            show_plot : Bool
                Calls plt.show() if True.
        """
        n_plots = self.__shape[2]
        figure, axes = plt.subplots(nrows=1, ncols=n_plots, figsize=(2 * 6.4, 4.8), constrained_layout=True)
        figure.suptitle(title)
        for index in range(n_plots):
            axes[index].imshow(self.__weights[:, :, index])
            if labels is None:
                axes[index].set_title(index)
            else:
                axes[index].set_title(labels[index])          
        if show_plot is True:
            plt.show()
