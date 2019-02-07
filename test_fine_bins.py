import sky as sk
import numpy as np
import util


def test_center_finder(filename, radius):
    sphere = util.load_data(filename)
    sky = sk.Sky(sphere, bin_space=100)
    centers = sky.vote(radius, error=sky.bin_space, blob_size=2, threshold=10)

    path = filename.split('.')[0] + '_' + str(radius)
    util.pickle_sky(sky, path)


def test_blob(filename, radius):
    sky = util.unpickle_sky(filename)
    if not isinstance(sky, sk.Sky):
        raise ValueError("Object is of type " + type(sky))
    filename = filename.split('/')[1]
    threshold = sky.get_threshold(radius)
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
parser.add_argument('filename', metavar='f', type=str, nargs=1)
parser.add_argument('radius', metavar='r', type=int, nargs=1)
args = parser.parse_args()
filename = args.filename[0]
radius = args.radius[0]
test_center_finder('Data/'+filename, radius)

