from centerfinder import util
from centerfinder import sky


if __name__ == '__main__':
    sky_ = sky.Sky(util.load_data('data/cf_mock_catalog_83C_120R_randoms_added.fits'), 5)
    # expected radius should be default to 108
    sky_.vote(radius=108)
    sky_.blobs_thres(radius=108, blob_size=3, type_='difference')
    sky_.plot_sky()
    # need to know centers_generated
    sky_.plot_eval(centers_generated=83)