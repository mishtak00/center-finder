import numpy as np
from astropy.io import fits
import pickle
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
        typ_ = hdul[1].data['WEIGHT_FKP']  # placeholder
        return np.vstack([ra, dec, z, typ_])


def test_vote(filename):
    for radius in range(90, 130, 3):
        # sky_ = sky.Sky(read_fits(filename), 5)
        sky_ = sky.Sky(util.load_data(filename), 5)
        sky_.vote(radius=radius)
        sky_.vote_2d_1d()
        path = filename.split('.')[-2].split('/')[-1]
        path = '../models/' + path + '_' + str(radius)
        print(path, filename)
        util.pickle_sky(sky_, path)

def test_blob(sky_, radius, rms_factor=1, filename=None):

    sky_.blobs_thres(radius=radius, blob_size=5, type_='difference', rms_factor=rms_factor)
    util.pickle_sky(sky_, filename)
    center_num = len(sky_.centers)
    ev = sky_.eval()
    eff = ev[1]
    center_f_true = ev[2]
    with open(filename+ '_rms.txt', 'w') as f:
        f.write(str(center_num))
        f.write('\n')
        f.write(str(rms_factor))
        f.write('\n')
        f.write(str(eff))
        f.write('\n')
        f.write(str(center_f_true))
        f.write('\n---------------------------------------\n')
    return [radius, rms_factor, center_num, eff, center_f_true]


def scan_rms(filename):
    lst = []
    sky_ = util.unpickle_sky(filename)
    if not isinstance(sky_, sky.Sky):
        raise ValueError("Object is of type " + type(sky_))
    for rms in np.arange(0.7, 0.9, 0.1):
        print(rms)
        tmp = test_blob(sky_, 108, rms_factor=rms, filename=filename)
        lst.append(tmp)
    lst = np.asarray(lst)
    with open(filename + '_rms_pickle', 'wb') as f:
        pickle.dump(lst, f)


def scan_radius(filename):
    lst = []
    for radius in range(90, 130, 3):
        fname = filename + '_' + str(radius)
        sky_ = util.unpickle_sky(fname)

        for rms in np.arange(1, 1.76, 0.25):
            tmp = test_blob(sky_, radius, rms_factor=rms, filename=fname)
            lst.append(tmp)
    lst = np.asarray(lst)
    with open(filename + 'set5_radius_pickle_0', 'wb') as f:
        pickle.dump(lst, f)

test_vote('../data/cf_set_5_mock_1.fits')
test_vote('../data/cf_set_5_mock_2.fits')
test_vote('../data/cf_set_5_mock_3.fits')
test_vote('../data/cf_set_5_mock_4.fits')
scan_radius('../models/cf_set_5_mock_1')
scan_radius('../models/cf_set_5_mock_2')
scan_radius('../models/cf_set_5_mock_3')
scan_radius('../models/cf_set_5_mock_4')


