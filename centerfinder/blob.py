import numpy as np
from skimage.feature import blob_dog


def dog(n_observed, n_expected, type_, blob_size, rms_factor):

    # if type_ == 'ratio':
    #     distr = n_observed.flatten()
    #     distr = np.nan_to_num(distr)
    #     rms = np.sqrt(np.nanmean(n_expected ** 2))
    #     factor = np.nanmean(n_expected) + rms * rms_factor
    #     blobs = blob_dog(n_observed / n_expected, min_sigma=1, max_sigma=blob_size, threshold=factor)
    #     return blobs[:, :3]
    # elif type_ == 'difference':
    significance = (n_observed - n_expected) / np.sqrt(n_expected)
    significance = np.nan_to_num(significance)

    # what the heck is this doing???
    # rms = np.sqrt(np.nanmean(n_expected ** 2))

    blobs = blob_dog(significance, min_sigma=1, max_sigma=blob_size, threshold=rms_factor)
    return np.asarray(blobs[:, :3].T, dtype=int)
