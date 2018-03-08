import utils
import os
import numpy as np

###############################################################
# class Dataset
# Defines the next_batch function for extracting mini-batches
###############################################################
class Dataset:
    def __init__(self, phase, num_categories, shuffle=False):
        # Load the complete modelnet40 training data
        # Training data is dispersed over 5 files,
        # load them all into data (point clouds) and labels
        points = []
        labels = []
        i = 0
        while True:
            filename = os.path.join("modelnet40", "ply_data_%s%i.h5" % (phase, i))
            if not os.path.isfile(filename): break
            p, l = utils.load_h5(filename)
            points.extend(np.array(p))
            labels.extend(np.squeeze(np.array(l), 1))
            i += 1

        self.num_examples = len(points)
        self._index_in_epoch = 0
        self._points = np.array(points)
        self._labels = np.zeros((self.num_examples, num_categories))
        self._labels[np.arange(self.num_examples), np.array(labels)] = 1
        self._shuffle = shuffle
        self._epoch_complete = False
        
        if self._shuffle:
            self._shuffle_data()
        
    def _shuffle_data(self):
        idx = np.arange(0, self.num_examples)
        np.random.shuffle(idx)
        self._points = self._points[idx]
        self._labels = self._labels[idx]

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        end = start + batch_size
        if end >= self.num_examples:
            end = self.num_examples
            self._index_in_epoch = 0
            self._epoch_complete = True
            
            if self._shuffle:
                self._shuffle_data()
        else:
            self._index_in_epoch = end
        return self._points[start:end], self._labels[start:end]
    
    def is_epoch_complete(self):
        if self._epoch_complete:
            self._epoch_complete = False
            return True
        else:
            return False

###############################################################
# class ModelNet
# Loads the modelnet traning and test set
###############################################################
class ModelNet:
    def __init__(self, shuffle=False):
        self.categories = utils.get_category_names()
        self.num_categories = len(self.categories)
        self.train = Dataset("train", self.num_categories, shuffle)
        self.test = Dataset("test", self.num_categories, shuffle)
        self.num_points = self.train._points.shape[1]
        
###############################################################
# def rotate
# Rotates one or a batch of point clouds along the up-axiz (y)
###############################################################
def rotate(points, theta):
    if points.ndim == 2: points = np.expand_dims(points, axis=0)   
    rotation_matrix = np.array([[ np.cos(theta), 0, np.sin(theta)],
                                [ 0,             1,             0],
                                [-np.sin(theta), 0, np.cos(theta)]])
    rotation_matrix = np.expand_dims(rotation_matrix, axis=0)
    rotation_matrix  = np.repeat(rotation_matrix, len(points), axis=0)
    return np.matmul(points, rotation_matrix)

###############################################################
# def rotate
# Jitter one or a batch of point clouds by adding
# gaussian noise to the xyz components
###############################################################
def jitter(points, mean, std):
    return points + np.random.normal(mean, std, points.shape)

###############################################################
# save/load objects with pickle
###############################################################
import pickle
def save_object(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
    return None

###############################################################
# linear interpolation
###############################################################
def lerp(a, b, f):
    return a + f * (b - a)