import os
from centerfinder import util
from centerfinder import sky


def test_pickle():
    sky_ = sky.Sky(util.load_data('data/cf_mock_catalog_83C_120R.fits'), 5)
    # expected radius should be default to 108
    filename = 'dummy'
    sky_.vote(radius=108)
    util.pickle_sky(sky_, filename)

    sky_1 = util.unpickle_sky(filename)
    sky_1.find_blob(radius=108, blob_size=3, type_='difference')
    os.remove(filename)


def test_blob():
    sky_ = sky.Sky(util.load_data('data/cf_mock_catalog_83C_120R.fits'), 5)
    # expected radius should be default to 108
    sky_.vote(radius=108)
    sky_.find_blob(radius=108, blob_size=3, type_='difference')
