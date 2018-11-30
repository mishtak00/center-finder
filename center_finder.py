import os.path as osp
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial
from matplotlib import rc


# import dask.array as da

class Sky:
    def __init__(self, xyz_list, avg_bins=100):
        if len(xyz_list) != 0:
            self.xyz_list = xyz_list
        else:
            raise ValueError('List of coordinates has wrong shape')
        self.ranges = []
        for i in range(3):
            self.ranges.append((min(xyz_list[i]), max(xyz_list[i])))
        self.bin_space = np.mean([i[1] - i[0] for i in self.ranges]) / avg_bins

        # TODO: ceiling
        self.bins = [math.ceil((self.ranges[i][1] - self.ranges[i][0]) / self.bin_space) for i in range(3)]
        print(self.bins)
        self.grid = np.zeros(self.bins)

    def coord_to_grid(self, point):
        index = np.array(
            [min(self.bins[i] - 1, int((point[i] - self.ranges[i][0]) / self.bin_space)) for i in range(3)])
        index[index < 0] = 0
        return index

    def grid_to_coord(self, point):
        index = np.array(
            [min(self.ranges[i][1], point[i] * self.bin_space + self.ranges[i][0]) for i in range(3)])
        return index

    def center_finder(self, radius, error=-1):
        if error == -1:
            error = self.bin_space

        for point in zip(*self.xyz_list):
            if np.isnan(point).any():
                pass
            else:
                sphere = draw_sphere(point, radius, self.bin_space, error)
                sphere = np.array([self.coord_to_grid(p) for p in sphere]).T
                self.grid[sphere[0], sphere[1], sphere[2]] += 1

        # threshold = self.grid.max() / 2
        sorted_grid = sorted(self.grid.flatten())
        threshold = sorted_grid[-5]
        print('threshold: ', threshold)
        ret = np.array(np.where(self.grid >= threshold)).T
        print(len(ret))
        return [self.grid_to_coord(r) for r in ret], threshold


def load_data(filename):
    # filename = osp.join(osp.abspath('../data'), filename)
    data = np.genfromtxt(filename, unpack=True).T
    ra = data[:, 0]
    dec = data[:, 1]
    z = data[:, 2]
    typ = data[:, 3]
    space = 1
    ra = ra[::space]
    dec = dec[::space]
    z = z[::space]
    typ = typ[::space]
    return ra, dec, z, typ


def SphereToSky(x, y, z, degrees=True):
    pass


def SkyToSphere(ra, dec, z, typ=0, degrees=True):
    if degrees:
        ra = np.radians(ra)
        dec = np.radians(dec)

    # cartesian coordinates
    x = np.cos(dec) * np.cos(ra) * z
    y = np.cos(dec) * np.sin(ra) * z
    z = np.sin(dec) * z
    # return np.stack([x,y,z]).T
    return x, y, z, typ


def center_finder_1(x, y, z, radius, bin_space=-1):
    if not len(x) == len(y) == len(z):
        raise ValueError("x y z length not equal")
    if bin_space == -1:
        bin_space = np.mean([max(col) - min(col) for col in [x, y, z]]) / 100
    print(bin_space)
    error = bin_space
    axis = []
    for col in [x, y, z]:
        bins = int((max(col) - min(col)) / bin_space)
        print(bins)
        axis.append(np.linspace(min(col), max(col), bins))
    x_grid, y_grid, z_grid = np.meshgrid(*axis)
    grid = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T
    print(grid.shape)
    kdtree = spatial.KDTree(grid)
    for x, y, z in zip(x, y, z):
        sphere = draw_sphere((x, y, z), radius, bin_space, error)
        # print(distance((x, y, z), sphere[0]))
        indexes = [kdtree.query(point, 2)[1] for point in sphere]
        threed_indexes = np.vstack([grid[i, :] for i in indexes])
        # print(threed_indexes.shape)


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
    diameter_bins = int(radius * 2 / bin_space)
    x, y, z = point[:3]
    sphere_coord_x = np.linspace(x - radius, x + radius, diameter_bins)
    sphere_coord_y = np.linspace(y - radius, y + radius, diameter_bins)
    sphere_coord_z = np.linspace(z - radius, z + radius, diameter_bins)
    # sphere = zip(sphere_coord_x, sphere_coord_y, sphere_coord_z)
    a, b, c = np.meshgrid(sphere_coord_x, sphere_coord_y, sphere_coord_z)
    sphere = zip(a.ravel(), b.ravel(), c.ravel())
    sphere = [point2 for point2 in sphere if radius - error < distance(point, point2) < radius + error]
    return sphere


def distance(pointA, pointB):
    x1, y1, z1 = pointA[:3]
    x2, y2, z2 = pointB[:3]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** .5


def test():
    sphere = SkyToSphere(*load_data('SignalN3.txt'))
    x, y, z = sphere
    print(sphere[0])
    # center_finder(*sphere, 0.1)
    print(len(sphere[0]), min(sphere[0]), max(sphere[0]), sphere[0].mean())
    print(len(sphere[1]), min(sphere[1]), max(sphere[1]), sphere[1].mean())
    print(len(sphere[2]), min(sphere[2]), max(sphere[2]), sphere[2].mean())
    ax = Axes3D(plt.gcf())
    ax.scatter(xs=sphere[0], ys=sphere[1], zs=sphere[2])
    sky = Sky(sphere)
    bin_space = np.mean([max(col) - min(col) for col in [x, y, z]]) / 100
    print(bin_space)
    error = bin_space / 100
    sph = draw_sphere((-0.575657871781208, -0.4043735011348877, 0.09756173221377211), 0.1,
                      bin_space=bin_space, error=error)
    sph = np.array(sph).T
    ax.scatter(xs=sph[0], ys=sph[1], zs=sph[2])
    print('sph: ', sph.shape)
    plt.show()


def test2():
    sphere = SkyToSphere(*load_data('SignalN3_easy.txt'))
    sky = Sky(sphere)
    centers = sky.center_finder(0.105)
    print(centers)
    print(len(centers))
    sphere = SkyToSphere(*load_data('SignalN3_easy.txt'))
    ax = Axes3D(plt.gcf())
    ax.scatter(xs=sphere[0], ys=sphere[1], zs=sphere[2])

    for i in centers:
        sph = draw_sphere(i, 0.1, sky.bin_space)
        sph = np.array(sph).T
        ax.scatter(xs=sph[0], ys=sph[1], zs=sph[2])
    centers = np.array(centers).T
    ax.scatter(centers[0], centers[1], centers[2], color='r')
    plt.show()


def test3():
    sphere = SkyToSphere(*load_data('SignalN3_easy.txt'))
    sky = Sky(sphere)
    sphere_stack = np.vstack(sphere).T
    print(sphere_stack.shape)
    centers = [point for point in sphere_stack if point[3] == 2]
    ax = Axes3D(plt.gcf())
    print(len(centers))
    for i in centers[:5]:
        sph = draw_sphere(i, 0.1, sky.bin_space, error=sky.bin_space / 10)
        sph = np.array(sph).T
        ax.scatter(xs=sph[0], ys=sph[1], zs=sph[2], alpha=0.05)
    centers = np.array(centers).T
    ax.scatter(centers[0], centers[1], centers[2], color='r')
    plt.show()


def test4():
    ax = Axes3D(plt.gcf())
    sphere = SkyToSphere(*load_data('SignalN3_easy.txt'))
    sky = Sky(sphere, avg_bins=50)
    sphere = SkyToSphere(*load_data('SignalN3_easy.txt'))
    print(len(sphere[3]))
    sphere_stack = np.vstack(sphere).T
    print(sphere_stack.shape)
    centers = [point for point in sphere_stack if point[3] == 2]
    ax.scatter(xs=sphere[0], ys=sphere[1], zs=sphere[2], color='blue', alpha=0.1)
    print(centers)
    centers = np.array(centers).T
    ax.scatter(centers[0], centers[1], centers[2], color='blue')
    radius = 0.04
    print('radius: ', radius)
    centers, threshold = sky.center_finder(radius)
    print('centers found: ', centers)

    print('threshold, length: ', threshold, len(centers))
    for i in centers:
        sph = draw_sphere(i, radius, sky.bin_space, error=sky.bin_space)
        sph = np.array(sph).T
        ax.scatter(xs=sph[0], ys=sph[1], zs=sph[2], alpha=0.05, color='r')
        ax.scatter(i[0], i[1], i[2], color='r')
    # plt.show()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend(['rim in data', 'center in data', 'rim found', 'center found'])
    plt.title(r'SignalN3 (easy)')
    # plt.savefig(osp.join('Figures', 'Figure_1130_1.png'))
    plt.show()


if __name__ == '__main__':
    test4()
