import numpy as np
from skimage.feature import blob_dog
import matplotlib.pyplot as plt


def dog(data, threshold, type_, blob_size, rms_factor):

    if type_ == 'ratio':
        distr = data.flatten()
        distr = np.nan_to_num(distr)
        rms = np.sqrt(np.nanmean(threshold ** 2))
        factor = np.nanmean(threshold) + rms * rms_factor
        blobs = blob_dog(data / threshold, min_sigma=1, max_sigma=blob_size, threshold=factor)
        return blobs[:, :3]
    elif type_ == 'difference':
        distr = (data - threshold) / np.sqrt(threshold)
        plt.imshow(distr[100])
        plt.show()
        distr = np.nan_to_num(distr)
        rms = np.sqrt(np.nanmean(threshold ** 2))
        blobs = blob_dog(distr, min_sigma=1, max_sigma=blob_size, threshold=rms_factor)
        return blobs[:, :3]
