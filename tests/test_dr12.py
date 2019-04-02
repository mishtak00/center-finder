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


test_vote('../data/galaxy_DR12v5_CMASS_South.fits')