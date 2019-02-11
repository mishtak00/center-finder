import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import norm
from centerfinder import sky as sk
from centerfinder import util


def plot_eval(sky, center_num):
    rc('font', family='serif')
    # rc('font', size=16)
    distr, center = sky.eval()
    mean, std = norm.fit(distr)
    num = len(np.where((distr >= mean - 3 * std) & (distr <= mean + 3 * std)))
    try:
        efficiency = len([x for x in distr if x < 18]) / center_num
        center_ratio = center/center_num
        fake = len([x for x in distr if x >= 18]) / len(distr)
        multiplicity = len([x for x in distr if x < 18]) / center
        return np.asarray([efficiency, fake, multiplicity, center_ratio])
    except:
        return np.asarray([0, 1, 0, 0])


def test_stat(filename, center_num, radius):
    sky = util.unpickle_sky(filename)
    if not isinstance(sky, sk.Sky):
        raise ValueError("Object is of type " + type(sky))
    threshold = sky.get_threshold(108)
    sky.centers = sky.blobs_thres(threshold, error=sky.bin_space, radius=radius, blob_size=3)
    return plot_eval(sky, center_num)

stat = []
filename = 'data/cf_mock_catalog_83C_120R_'
for r in range(96, 130, 3):
    stat.append(test_stat('data/cf_mock_catalog_83C_120R_'+str(r), 83, radius=r))
with open('stats', 'wb') as f:
    pickle.dump(stat, f)
print(stat)

with open('stats', 'rb') as f:
    arr = pickle.load(f)
arr = np.vstack(arr).T
print(arr)

rc('font', family='serif')
#rc('font', size=16)
rs = range(96, 130, 3)
print(len(rs))
plt.figure(figsize=(6, 5))
plt.plot(rs, arr[0])
plt.plot(rs, arr[1])
plt.plot(rs, arr[2])
plt.xlabel('radius')
plt.legend(['efficiency', 'fake rate', 'multiplicity'], loc='upper left')
plt.title('Mock with nrim=120, center=83')
plt.tight_layout()
#plt.show()
plt.savefig('Figures/Figure_cf_mock_catalog_83C_120R_stats.png')

