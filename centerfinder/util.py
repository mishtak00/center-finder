import pickle
import sys
from typing import List

import math
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve

from . import sky

sys.modules['sky'] = sky


def load_data(filename: str) -> np.ndarray:
    """
    load the data from file; can take in both plain text and .fits
    :param filename:
    :param space:
    :return:
    """
    if filename.split('.')[-1] == 'fits':
        hdul = fits.open(filename)
        ra = hdul[1].data['ra']
        dec = hdul[1].data['dec']
        z = hdul[1].data['z']
        typ_ = hdul[1].data['weight']
        return np.vstack([ra, dec, z, typ_])

    data = np.genfromtxt(filename, unpack=True).T
    ra = data[:, 0]
    dec = data[:, 1]
    z = data[:, 2]
    typ_ = data[:, 3]
    return np.vstack([ra, dec, z, typ_])


def local_thres(data: np.ndarray, region: [int, float]) -> np.ndarray:
    """
    Mean-filter the 3-d grid
    :param data:
    :param region:
    :return:
    """
    region = int(region)
    w = np.full((region, region, region), 1.0 / (region ** 3))
    ret = fftconvolve(data, w, mode='same')
    return ret


def sky_to_cartesian(points: np.ndarray, degrees=True):
    """

    :param points:
    :param typ:
    :param degrees:
    :return:
    """
    if len(points) < 3:
        raise ValueError('Input dimension should be >= 3; get {:d}'.format(len(points)))
    ra, dec, z = points[:3]
    if degrees:
        ra = np.radians(ra)
        dec = np.radians(dec)

    omegaM = 0.274
    norm = 3000
    func = z * (1 - omegaM * 3 * z / 4)
    r = norm * func

    x = np.cos(dec) * np.cos(ra) * r
    y = np.cos(dec) * np.sin(ra) * r
    z = np.sin(dec) * r
    if len(points) == 4:
        typ = points[3]
        return np.vstack([x, y, z, typ])
    return np.vstack([x, y, z])


def cartesian_to_sky(points: np.ndarray):
    """

    :param points:
    :return: ndarray in shape (3, *)
    """
    if len(points) < 3:
        raise ValueError('Input dimension should be > 3; get {:d}'.format(len(points)))
    x, y, z = points[:3]
    s = np.sqrt(x ** 2 + y ** 2)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, s)

    ra = np.degrees(lon)
    ra = (ra + 360) % 360
    dec = np.degrees(lat)

    omegaM = 0.274
    norm = 3000
    r = z / np.sin(np.radians(dec))
    func = r / norm
    z = 2.43309 - 0.108811 * np.sqrt(500 - 411 * func)

    return np.vstack([ra, dec, z])


def distance_kernel(radius: [float, int], bin_space: [float, int], error=-1):
    """
    :param radius: distance
    :param bin_space:
    :param error:
    :return:
    """
    if error == -1:
        error = bin_space / 2
    outer_bins = int((radius + error * 5) / bin_space)
    outer_r = int((radius + error) / bin_space)
    inner_bins = int((radius - error) / bin_space)
    xyz = [np.arange(outer_bins * 2 - 1) for i in range(3)]
    window = np.vstack(np.meshgrid(*xyz)).reshape(3, -1)

    center = [int(outer_bins) - 1, int(outer_bins) - 1, int(outer_bins) - 1]
    dist = np.asarray(distance(center, window))
    dist[dist < outer_bins] = -1
    dist[dist > 0] = 0
    dist[dist < 0] = 1

    dist = dist.reshape((outer_bins * 2 - 1, outer_bins * 2 - 1, outer_bins * 2 - 1))
    return dist


def kernel(radius: [float, int], bin_space: [float, int], error=-1):
    """
    The convolution kernel
    :param radius:
    :param bin_space:
    :param error:
    :return:
    """
    if error == -1:
        error = bin_space
    outer_bins = int((radius + error * 5) / bin_space)
    outer_r = int((radius + error) / bin_space)
    inner_bins = int((radius - error) / bin_space)
    xyz = [np.arange(outer_bins * 2 - 1) for i in range(3)]
    window = np.vstack(np.meshgrid(*xyz)).reshape(3, -1)

    center = [int(outer_bins) - 1, int(outer_bins) - 1, int(outer_bins) - 1]
    kernel = np.asarray(distance(center, window))
    kernel[kernel < 2] = np.inf
    kernel[kernel < inner_bins] = 0
    kernel[kernel == np.inf] = -10
    kernel[kernel > outer_r] = 0
    kernel[kernel > 0] = 1
    kernel[kernel == -10] = 0

    kernel = kernel.reshape((outer_bins * 2 - 1, outer_bins * 2 - 1, outer_bins * 2 - 1))

    return kernel


def draw_sphere(point, radius, bin_space, error=-1):
    """

    :param point:
    :param radius:
    :param bin_space:
    :param error:
    :return: list of point coordinates that are on the surface of the sphere
    """
    if error == -1:
        error = bin_space
    diameter_bins = int((radius + error) * 2 / bin_space)
    # diameter_bins = 20
    x, y, z = point[:3]
    sphere_coord_x = np.linspace(x - radius - error, x + radius + error, diameter_bins)
    sphere_coord_y = np.linspace(y - radius - error, y + radius + error, diameter_bins)
    sphere_coord_z = np.linspace(z - radius - error, z + radius + error, diameter_bins)
    # sphere = zip(sphere_coord_x, sphere_coord_y, sphere_coord_z)
    a, b, c = np.meshgrid(sphere_coord_x, sphere_coord_y, sphere_coord_z)
    sphere = zip(a.ravel(), b.ravel(), c.ravel())
    sphere = [point2 for point2 in sphere if radius - error < distance(point, point2) < radius + error]
    return sphere


def distance(point_a: List, point_b: List):
    """
    Distance between 2 points; can pass in numpy arrays
    :param point_a:
    :param point_b:
    :return:
    """
    if len(point_a) < 3 or len(point_b) < 3:
        raise ValueError('Input dimension should be > 3; get {:d} and {:d}'.format(len(point_a), len(point_b)))
    x1, y1, z1 = point_a[:3]
    x2, y2, z2 = point_b[:3]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** .5


def sphere_fit(sp_x, sp_y, sp_z):
    """
    Fit 3-d sphere
    :param sp_x:
    :param sp_y:
    :param sp_z:
    :return:
    """
    #   Assemble the A matrix
    sp_x = np.array(sp_x)
    sp_y = np.array(sp_y)
    sp_z = np.array(sp_z)
    A = np.zeros((len(sp_x), 4))
    A[:, 0] = sp_x * 2
    A[:, 1] = sp_y * 2
    A[:, 2] = sp_z * 2
    A[:, 3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(sp_x), 1))
    f[:, 0] = (sp_x * sp_x) + (sp_y * sp_y) + (sp_z * sp_z)
    C, residules, rank, singval = np.linalg.lstsq(A, f)

    #   solve for the radius
    t = (C[0] * C[0]) + (C[1] * C[1]) + (C[2] * C[2]) + C[3]
    radius = math.sqrt(t)

    return radius, [*C[0], C[1], C[2]]


def pickle_sky(sky_: sky, filename):
    with open(filename, 'wb') as f:
        pickle.dump(sky_, f)
    return 0


def unpickle_sky(filename):
    with open(filename, 'rb') as f:
        sky_ = pickle.load(f)
    return sky_


def plot_cross_sec(data: np.ndarray, thres_grid: np.ndarray = None, c_found=None, c_generated=None):
    if data.ndim != 3:
        raise ValueError('Grid to plot should be 3-dimensional')
    section = int(len(data) / 9)
    if 3 <= c_found.shape[1] <= 4:
        c_found = c_found.T
    if 3 <= c_generated.shape[1] <= 4:
        c_generated = c_generated.T

    # data -= thres_grid
    data1 = np.copy(data)
    vmin = np.min(np.nan_to_num(data))
    vmax = np.max(np.nan_to_num(data))
    # vmin = np.percentile(np.nan_to_num(data), 1)
    # vmax = np.percentile(np.nan_to_num(data), 99)
    print(vmin, vmax)
    data[c_found[0], c_found[1], c_found[2]] = vmax * 1.5
    # data[c_found[0], c_found[1], c_found[2]-1] = vmax * 1.5
    # data[c_found[0], c_found[1], c_found[2]+1] = -5
    # data = local_thres(data, 2)
    #
    data1[c_generated[0], c_generated[1], c_generated[2]] = vmax * 1.5
    # data1[c_generated[0], c_generated[1], c_generated[2]-1] = vmax * 1.5
    # data1[c_generated[0], c_generated[1], c_generated[2]+1] = -5
    # data1 = local_thres(data1, 2)
    vmax *= 1.2

    f, axarr = plt.subplots(2, 3)
    im = axarr[0, 0].imshow(data[:, :, section * 3], vmin=vmin, vmax=vmax)
    im = axarr[1, 0].imshow(data1[:, :, section * 3], vmin=vmin, vmax=vmax)
    im = axarr[0, 1].imshow(data[:, :, section * 4], vmin=vmin, vmax=vmax)
    im = axarr[1, 1].imshow(data1[:, :, section * 4], vmin=vmin, vmax=vmax)
    im = axarr[0, 2].imshow(data[:, :, section * 5], vmin=vmin, vmax=vmax)
    im = axarr[1, 2].imshow(data1[:, :, section * 5], vmin=vmin, vmax=vmax)
    # cols = ['N-obs', 'N-exp', 'N-obs/N-exp']
    # f.colorbar(im, ax=axarr.ravel().tolist())
    # for ax, col in zip(axarr[0], cols):
    #     ax.set_title(col)

    # plt.colorbar(im, ax=axarr.ravel().tolist())
    plt.tight_layout()
    plt.show()


def conv(data, window):
    return fftconvolve(data, window, mode='same')
