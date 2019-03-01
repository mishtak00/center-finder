import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
from typing import Callable
from . import sky
from . import util


def chi_sqr_kernel(radius: [float, int], bin_space: [float, int], error=-1) -> np.ndarray:
    if error == -1:
        error = bin_space
    outer_bins = int((radius + error * 1) / bin_space)
    radius_bins = radius / bin_space
    xyz = [np.arange(outer_bins * 2 - 1) for i in range(3)]
    window = np.vstack(np.meshgrid(*xyz)).reshape(3, -1)

    center = [int(outer_bins) - 1, int(outer_bins) - 1, int(outer_bins) - 1]
    dist = np.asarray(util.distance(center, window))
    dist = (dist - radius_bins) ** 2
    dist = dist.reshape((outer_bins * 2 - 1, outer_bins * 2 - 1, outer_bins * 2 - 1))
    dist [dist > 1*error] = 0
    return dist


def plot_over_radius(filename: str, center_num: int, func: Callable, radius_step=3, bin_space=5) -> None:
    val_list = []
    for r in range(90, 130, radius_step):
        sky_ = sky.Sky(util.load_data(filename), bin_space)
        sky_.find_center(r, 3, 'difference')
        val_list.append(func(sky_, center_num))
    rc('font', family='serif')
    func_name = func.__name__
    func_name = func_name.replace('_', ' ')
    rs = range(90, 130, radius_step)
    # plt.figure(figsize=(6, 5))
    # rc('font', size=16)
    plt.plot(rs, val_list)
    plt.xlabel('radius')
    plt.ylabel(func_name)
    # plt.legend(['efficiency', 'fake rate', 'multiplicity'], loc='upper left')
    plt.title(filename.split('/')[-1].split('.')[0])
    plt.tight_layout()
    plt.show()


def efficiency(sky_: sky.Sky, center_num: int) -> float:
    distr, center = sky_.eval()
    efficiency = len([x for x in distr if x < 18]) / center_num
    return efficiency


def fake_rate(sky_: sky.Sky, center_num: int = 0) -> float:
    distr, center = sky_.eval()
    try:
        fake_rate = len([x for x in distr if x >= 18]) / len(distr)
    except ZeroDivisionError:
        fake_rate = 1
    return fake_rate


def centers_found(sky_: sky.Sky, center_num: int = 0) -> int:
    distr, center = sky_.eval()
    return len([x for x in distr if x < 18])


def chi_sqr(sky_: sky.Sky, radius, true_center=True):
    grid = np.zeros(sky_.grid.shape)
    galaxies = sky_._coord_to_grid(sky_.xyz_list)
    grid[galaxies[0], galaxies[1], galaxies[2]] += 1
    gaus = util.local_thres(sky_.get_center_grid(), 18 / sky_.space_3d)
    confirmed = []
    fake = []
    for c_grid in sky_.centers:
        c_grid = sky_._coord_to_grid(c_grid)
        c_grid = [int(c) for c in c_grid]
        if gaus[c_grid[0], c_grid[1], c_grid[2]] > 0:
            confirmed.append(c_grid)
        else:
            fake.append(c_grid)
    if true_center:
        c_list = np.asarray(confirmed).T
    else:
        c_list = np.asarray(fake).T
    grid = util.conv(grid, chi_sqr_kernel(radius, sky_.space_3d))
    val_list = grid[c_list[0], c_list[1], c_list[2]]
    return val_list

def plot_chi_sqr(sky_: sky.Sky, radius):
    true_list = chi_sqr(sky_, radius)
    fake_list = chi_sqr(sky_, radius, true_center=False)
    plt.hist([true_list, fake_list], bins=20, density=False)
    plt.xlabel('Chi-square')
    plt.legend(['true centers', 'fake centers'])
    plt.show()
