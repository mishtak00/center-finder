import sky as sk
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import util
from scipy.stats import norm


def test_center_finder(filename, radius):

    sphere = util.load_data(filename)
    sky = sk.Sky(sphere, avg_bins=20)
    centers = sky.center_finder(radius, error=sky.bin_space, blob_size=2, threshold=10)

    print('------------------------------')
    path = 'test_easy'
    util.pickle_sky(sky, 'Data/' + path)


def test_blob(filename):
    sky = util.unpickle_sky(filename)
    if not isinstance(sky, sk.Sky):
        raise ValueError("Object is of type " + type(sky))
    threshold = sky.get_threshold(108)
    print(threshold)
    util.plot_threshold(threshold)
    sky.centers = sky.blobs_thres(threshold, error=sky.bin_space, radius=108, blob_size=2)
    sky.plot_sky(show_rim=False, radius=108)
    print('threshold: ', np.mean(threshold), np.median(threshold), np.max(threshold))
    print(np.mean(sky.grid), np.median(sky.grid), np.max(sky.grid))
    ratio = (sky.grid / threshold).flatten()

    mean, std = norm.fit(ratio)
    plt.hist(ratio, bins=50, density=True)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, mean, std)
    plt.plot(x, y)
    # plt.show()

    #
    # plt.imshow(sky.grid[1])
    # plt.colorbar()
    # plt.show()
    # plt.imshow(sky.grid[5])
    # plt.colorbar()
    # plt.show()
    # plt.imshow(sky.grid[9])
    # plt.colorbar()
    # plt.show()
    # sky.grid += threshold
    # path = 'test_easy'
    # util.pickle_sky(sky, 'Data/' + path)

def test_grid(filename):
    sky = util.unpickle_sky(filename)
    if not isinstance(sky, sk.Sky):
        raise ValueError("Object is of type " + type(sky))
    util.plot_threshold(sky.grid)
    ratio = sky.grid.flatten()

    mean, std = norm.fit(ratio)
    plt.hist(ratio, bins=50, density=True)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, mean, std)
    plt.plot(x, y)
    plt.show()

def test_stat(filename):
    sky = util.unpickle_sky(filename)
    if not isinstance(sky, sk.Sky):
        raise ValueError("Object is of type " + type(sky))
    # plt.imshow(sky.grid_2d)
    # plt.show()
    print(np.mean(sky.grid), np.median(sky.grid), np.max(sky.grid))
    print('_______________________')
    print(len(sky.centers))
    sky.plot_eval()
    sky.plot_sky(show_rim=False, radius=108)


# test_center_finder('Data/cf_mock_catalog_83C_120R.fits', 108)
# test_grid('Data/test_easy')
test_blob('Data/cf_mock_catalog_83C_120R')
# test_stat('Data/test_easy')
