import sys
sys.path.append('.')
from centerfinder import sky
from centerfinder import util
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from typing import List
from matplotlib import rc
import pickle

from scipy.ndimage import gaussian_filter

def preprocess(filename: str):
    filename = filename.split('.')[0] + '.fits'
    sky_ = sky.Sky(util.load_data('../data/' + filename), bin_space=5, force_size=250)
    print(sky_.grid.shape)
    sky_.vote(radius=108)
    print(sky_.grid.shape)
    center_grid = sky_.get_center_grid() * 1e6
    center_grid = gaussian_filter(center_grid, 2)
    # plt.imshow(center_grid[100])
    # plt.show()
    filename = filename.split('.')[0] + '_train'
    filename = '../models/'+filename.split('/')[-1]
    with open(filename, 'wb') as f:
        pickle.dump([sky_.grid, center_grid], f)


preprocess('cf_mock_catalog_333C_30R.fits')
