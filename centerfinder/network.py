import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from typing import List
from matplotlib import rc
import pickle
from centerfinder import sky
from centerfinder import util

import os

os.environ['KERAS_BACKEND'] = 'theano'

from keras import backend as K

from keras.models import Sequential
from keras.layers import Convolution3D as Conv3D, MaxPooling3D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils


class Net:

    def __init__(self, width):
        # samples = 20
        # width = 250
        # X_train = np.random.rand(1, width, width, width, 1)
        # y_train = np.random.rand(1, width, width, width, 1)

        self.model = Sequential()
        self.model.add(Conv3D(filters=1,
                              kernel_size=(51, 51, 51),
                              activation="relu",
                              padding='same',
                              data_format="channels_last",
                              kernel_initializer=my_init,
                              input_shape=(width, width, width, 1)))
        self.model.summary()

    def train(self):
        pass

    def save(self, filename: str):
        model_json = self.model.to_json()
        with open("../models/" + filename + '.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("../models/" + filename + '.h5')
        print("Model saved")


def my_init(shape, dtype=None):
    kernel = util.sphere_window(108, 5)[:, :, :, None, None]
    print(kernel.shape)
    if kernel.shape != shape:
        raise ValueError('Cannot make kernel in shape ' + str(shape))
    return kernel

def loss(yTrue: np.ndarray, yPred: np.ndarray):
    if yTrue.shape != yPred.shape:
        raise ValueError('incompatible shapes')


def load_data(filename: str):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data[0][None, :, :, :, None], data[1][None, :, :, :, None]


filename = '../models/cf_mock_catalog_333C_30R_train'
X_train, Y_train = load_data(filename)
net = Net(250)
net.model.compile(loss='mse', optimizer='adam')
print(X_train.shape, Y_train.shape)
net.model.fit(X_train, Y_train)
net.save(filename + '_model')
