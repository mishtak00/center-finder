import sky as sk
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import util


def test_center_finder(filename, radius):
    sphere = util.load_data(filename)
    sky = sk.Sky(sphere, avg_bins=10)
    sky.plot_original()
    centers = sky.center_finder(radius, error=sky.bin_space, blob_size=3)
    print(centers)
    path = filename.split('.')[0]
    util.pickle_sky(sky, 'Data/' + path)


def test_eval(filename):
    sky = util.unpickle_sky(filename)
    if not isinstance(sky, sk.Sky):
        raise ValueError("Object is of type " + type(sky))
    print(sky.centers)
    sky.plot_eval()
    sky.plot_sky(show_rim=False, radius=105)


# test_center_finder('SignalN3_mid.txt', 105)
# test_eval(osp.join('Data', 'SignalN3_mid'))
test_center_finder('SignalN3_easy.txt', 105)
test_eval(osp.join('Data', 'SignalN3_easy'))

