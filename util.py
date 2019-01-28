import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
import pickle
from scipy.ndimage.filters import convolve


def load_data(filename, space=1):
    data = np.genfromtxt(filename, unpack=True).T
    ra = data[:, 0]
    dec = data[:, 1]
    z = data[:, 2]
    typ = data[:, 3]
    ra = ra[::space]
    dec = dec[::space]
    z = z[::space]
    typ = typ[::space]
    return ra, dec, z, typ

def local_thres(data, region):
    w = np.full((region, region, region), 1.0/(region ** 3))
    return convolve(data, w)


def SkyToSphere(ra, dec, z, typ=0, degrees=True):
    # if degrees:
    #     ra = np.radians(ra)
    #     dec = np.radians(dec)
    #
    # x = np.cos(dec) * np.cos(ra) * z
    # y = np.cos(dec) * np.sin(ra) * z
    # z = np.sin(dec) * z

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
    return x, y, z, typ

def CartesianToSky(x, y, z):
    '''
    :param x:
    :param y:
    :param z:
    :return: ra, dec, r
    TODO: currently returns r, not z
    '''
    s = np.sqrt(x **2 + y **2)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, s)

    ra = np.degrees(lon)
    ra = (ra + 360) % 360
    dec = np.degrees(lat)

    omegaM = 0.274
    norm = 3000
    r = z / np.sin(np.radians(dec))
    func = r / norm
    z = 2.43309 - 0.108811 * np.sqrt(500 - 411*func)

    return ra, dec, z




def sphere_window(radius, bin_space, error=-1):
    if error == -1:
        error = bin_space
    outer_bins = int((radius + error) / bin_space)
    inner_bins = int((radius - error) / bin_space)
    window = np.zeros((outer_bins * 2 + 1, outer_bins * 2 + 1, outer_bins * 2 + 1))

    center = int(outer_bins), int(outer_bins), int(outer_bins)
    for index, val in np.ndenumerate(window):
        if inner_bins <= distance(center, index) <= outer_bins:
            window[index] = 1
    return window, center


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


def distance(pointA, pointB):
    x1, y1, z1 = pointA[:3]
    x2, y2, z2 = pointB[:3]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** .5
    # return np.sum(pointA[:3] - pointB[:3]) ** .5


def plot_threshold(matrix, num):
    '''
    Self-defined plotting function for matrix matrices
    Decides the proper vmax and vmin when plotting matrix
    :param matrix: matrix matrix
    :param extent: the same 'extent' parameter as in imshow()
    :param title: the same 'title' parameter as in imshow()
    :return:
    '''
    matrix_sorted = matrix.flatten()
    matrix_sorted = sorted(matrix_sorted)
    first = int(len(matrix_sorted) / num)
    last = first * (num - 1)
    rc('font', family='serif')
    rc('font', size=16)
    plt.figure(figsize=[6, 6])
    plt.tight_layout()
    plt.imshow(matrix, vmax=matrix_sorted[last], vmin=matrix_sorted[first])
    plt.colorbar()
    plt.gca().invert_yaxis()


def sphere_fit(spX, spY, spZ):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX), 4))
    A[:, 0] = spX * 2
    A[:, 1] = spY * 2
    A[:, 2] = spZ * 2
    A[:, 3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX), 1))
    f[:, 0] = (spX * spX) + (spY * spY) + (spZ * spZ)
    C, residules, rank, singval = np.linalg.lstsq(A, f)

    #   solve for the radius
    t = (C[0] * C[0]) + (C[1] * C[1]) + (C[2] * C[2]) + C[3]
    radius = math.sqrt(t)

    return radius, [*C[0], C[1], C[2]]


def pickle_sky(sky, filename):
    with open(filename, 'wb') as f:
        pickle.dump(sky, f)
    return 0


def unpickle_sky(filename):
    with open(filename, 'rb') as f:
        sky = pickle.load(f)
    return sky


def plot_threshold(threshold, above=0):
    ax = Axes3D(plt.gcf())
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if above == 0:
        sorted = np.sort(threshold)
        above = sorted[int(9 * len(sorted) / 10)]

        for i in range(3, 20):
            above = np.median(threshold) * i
            centers_d = np.where(threshold >= above)
            ax.scatter(centers_d[0], centers_d[1], centers_d[2], color='blue', alpha=0.1)

    # self.grid[blob_idx] = 0


    plt.show()