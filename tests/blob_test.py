from centerfinder import util
from centerfinder import sky


if __name__ == '__main__':
    sky_ = sky.Sky(util.load_data('../data/cf_mock_catalog_83C_120R.fits'), 5)
    # expected radius should be default to 108
    # for mocks with high n-rim and low n-center, set rms_factor higher
    sky_.find_center(radius=108, blob_size=3, type_='difference', rms_factor=2.5)
    # need to know centers_generated from the mock
    sky_.plot_eval(centers_generated=83)
