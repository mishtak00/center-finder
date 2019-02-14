import time
import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt
from centerfinder import util
from centerfinder import sky


def stat(sky_: sky.Sky, centers_generated: int, radius: int):

    sky_.blobs_thres(radius, blob_size=3)
    distribution, true_centers = sky_.eval()
    found_eff = len([d for d in distribution if d <= 18])
    try:
        efficiency = found_eff / centers_generated
        fake_rate = (len(sky_.centers) - found_eff) / len(sky_.centers)
        multiplicity = found_eff / true_centers

        return np.asarray([efficiency, fake_rate, multiplicity])
    except:
        return np.asarray([0, 1, 0])


def meta_stat(filename: str, center_num: int) -> None:
    stat_ = []
    for r in range(96, 130, 3):
        sky_ = util.unpickle_sky(filename + '_' + str(r))
        stat_.append(stat(sky_, center_num, r))
    with open('../stats', 'wb') as f:
        pickle.dump(stat_, f)


def meta_plot(filename: str) -> None:
    with open(filename, 'rb') as f:
        stat_ = pickle.load(f)
    r = range(96, 130, 3)
    stat_ = np.vstack(stat_).T
    plt.plot(r, stat_[0])
    plt.plot(r, stat_[1])
    plt.plot(r, stat_[2])
    plt.legend(['efficiency', 'fake rate', 'multiplicity'], loc='upper right')
    plt.title('mock with nrim=30, ncenter=333')
    time = str(datetime.datetime.now())
    plt.savefig('../Figures/Figure_'+time+'_stats.png')
    plt.show()


if __name__ == '__main__':
    meta_stat('../models/cf_mock_catalog_333C_30R', 333)
    meta_plot('../stats')
