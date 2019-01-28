import sky as sk
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import util


def test_center_finder(filename, radius):
    sphere = util.load_data(filename)
    sky = sk.Sky(sphere, bin_space=5)
    centers = sky.center_finder(radius, error=sky.bin_space, blob_size=2, threshold=10)

    path = 'mid_fine_bins'
    util.pickle_sky(sky, 'Data/' + path)


def test_blob(filename):
    sky = util.unpickle_sky(filename)
    if not isinstance(sky, sk.Sky):
        raise ValueError("Object is of type " + type(sky))
    threshold = sky.get_threshold()
    # util.plot_threshold(threshold)
    sky.centers = sky.blobs(threshold, error=sky.bin_space, radius=105, blob_size=2)
    sky.plot_sky(show_rim=False, radius=105)
    print('threshold: ', np.mean(threshold), np.median(threshold), np.max(threshold))
    print(np.mean(sky.grid), np.median(sky.grid), np.max(sky.grid))

    plt.imshow(sky.grid[1])
    plt.colorbar()
    plt.show()
    plt.imshow(sky.grid[5])
    plt.colorbar()
    plt.show()
    plt.imshow(sky.grid[9])
    plt.colorbar()
    plt.show()
    sky.grid += threshold
    path = 'test_easy'
    util.pickle_sky(sky, 'Data/' + path)


def test_stat(filename):
    sky = util.unpickle_sky(filename)
    if not isinstance(sky, sk.Sky):
        raise ValueError("Object is of type " + type(sky))
    # plt.imshow(sky.grid_2d)
    # plt.show()
    sky.plot_original()
    print(np.mean(sky.grid), np.median(sky.grid), np.max(sky.grid))
    print('_______________________')
    sky.get_threshold()
    print(len(sky.centers))
    sky.plot_eval()
    sky.plot_sky(show_rim=False, radius=105)


filename = 'mid_fine_bins'
test_center_finder('SignalN3_mid.txt', 105)
test_blob('Data/' + filename)
test_stat('Data' + filename)
