import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from centerfinder import util
from centerfinder import blob
from centerfinder import sky


def read_fits(filename):
    if filename.split('.')[-1] == 'fits':
        hdul = fits.open(filename)
        ra = hdul[1].data['RA']
        dec = hdul[1].data['DEC']
        z = hdul[1].data['Z']
        typ_ = hdul[1].data['WEIGHT_FKP'] # placeholder
        return np.vstack([ra, dec, z, typ_])


def test_vote(filename):

    for radius in range(96, 120, 3):
        sky_ = sky.Sky(read_fits(filename), 5)
        # sky_.find_center(radius=108, blob_size=3, type_='diff')
        sky_.vote(radius=radius)
        path = filename.split('.')[-2].split('/')[-1]
        path = '../models/' + path + '_' + str(radius)
        print(path, filename)
        util.pickle_sky(sky_, path)


def test_blob(filename, radius):
    sky_ = util.unpickle_sky(filename)
    if not isinstance(sky_, sky.Sky):
        raise ValueError("Object is of type " + type(sky_))
    sky_.blobs_thres(radius=radius, blob_size=3, type_='difference')
    util.pickle_sky(sky_, filename)
    center_num = len(sky_.centers)
    with open('cmass_south_stat.txt', 'w+') as f:
        f.write(str(radius))
        f.write('\n')
        f.write(str(center_num))
        f.write('\n')
        f.write(str(radius))
        f.write('\n---------------------------------------\n')


def scan_radius(filename):
    for radius in range(96, 120, 3):
        fname = filename + '_' + str(radius)
        test_blob(fname, radius)


scan_radius('../models/galaxy_DR12v5_CMASS_South')