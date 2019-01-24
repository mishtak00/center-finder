import sky as sk
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import util


def test_center_finder(filename, radius):

    sphere = util.load_data(filename)
    sky = sk.Sky(sphere, avg_bins=50)
    centers = sky.center_finder(radius, error=sky.bin_space, blob_size=2, threshold=10)

    path = 'test_mid'
    util.pickle_sky(sky, 'Data/' + path)


def test_blob(filename):
    sky = util.unpickle_sky(filename)
    if not isinstance(sky, sk.Sky):
        raise ValueError("Object is of type " + type(sky))
    threshold = sky.get_threshold()
    # util.plot_threshold(threshold)
    centers = sky.blobs(threshold, error=sky.bin_space, radius=105, blob_size=4)
    sky.plot_sky(show_rim=False)
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
    path = 'test_mid'
    util.pickle_sky(sky, 'Data/' + path)


def test_stat(filename):
    sky = util.unpickle_sky(filename)
    if not isinstance(sky, sk.Sky):
        raise ValueError("Object is of type " + type(sky))
    # plt.imshow(sky.grid_2d)
    # plt.show()
    print(np.mean(sky.grid), np.median(sky.grid), np.max(sky.grid))
    print('_______________________')
    sky.get_threshold()
    print(len(sky.centers))
    sky.plot_eval()
    sky.plot_sky(show_rim=False, radius=105)


# test_center_finder('SignalN3_mid.txt', 105)

# data_list = util.load_data('SignalN3_easy.txt')
# xyz_list = util.SkyToSphere(*data_list)
# back = util.CartesianToSky(*xyz_list[:3])
# print(np.mean(data_list[0]), np.mean(data_list[1]), np.median(data_list[2]))
# print(np.mean(back[0]), np.mean(back[1]), np.median(back[2]))
# print(np.vstack(back).T)

test_blob('Data/test_mid')
test_stat('Data/test_mid')