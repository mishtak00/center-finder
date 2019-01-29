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

    path = 'mid_fine_bins_'+ str(radius)
    util.pickle_sky(sky, 'Data/' + path)


def test_blob(filename, radius):
    sky = util.unpickle_sky(filename)
    if not isinstance(sky, sk.Sky):
        raise ValueError("Object is of type " + type(sky))
    filename = filename.replace('/', '_')
    threshold = sky.get_threshold()
    # util.plot_threshold(threshold)
    sky.centers = sky.blobs(threshold, error=sky.bin_space, radius=radius, blob_size=2)
    sky.plot_sky(show_rim=False, radius=radius, savelabel=filename)
    print('threshold: ', np.mean(threshold), np.median(threshold), np.max(threshold))
    print(np.mean(sky.grid), np.median(sky.grid), np.max(sky.grid))

    sky.plot_eval(savelabel=filename)


def test_stat(filename, radius):
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
    sky.plot_sky(show_rim=False, radius=radius)


from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('radius', metavar='radius', type=int, nargs=1)
args = parser.parse_args()
radius = args.radius[0]
filename = 'mid_fine_bins_' + str(radius)
test_center_finder('SignalN3_mid.txt', radius)
test_blob('Data/' + filename, radius=radius)
#test_stat('Data/' + filename)
