import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from .blob import simple_maxima, dog
from scipy.stats import norm
from typing import List
from . import util


class Sky:
    def __init__(self, data_list: List, avg_bins=100, bin_space=0, force_size=None) -> None:
        """
        :param data_list: required
        :param self.avg_bins: if bin_space is specified, self.avg_bins will be overwritten
        :param bin_space: identical for all 3 dimensions
        """
        if len(data_list) >= 3:
            self.data_list = data_list
        else:
            raise ValueError('data list has wrong dimension')
        self.centers = []
        self.xyz_list = util.sky_to_cartesian(data_list)
        self.points_arr = np.vstack(self.xyz_list).T
        self.avg_bins = avg_bins
        ''' initialize 3d bins'''
        self.ranges = []
        for i in range(3):
            self.ranges.append((min(self.xyz_list[i]), max(self.xyz_list[i])))
        self.bin_space = np.mean([i[1] - i[0] for i in self.ranges]) / self.avg_bins

        if bin_space:
            self.bin_space = bin_space


        self.bins = [math.ceil((self.ranges[i][1] - self.ranges[i][0]) / self.bin_space) for i in range(3)]
        self.avg_bins = np.mean(self.bins)



        if force_size:
            self.bins = [force_size, force_size, force_size]
        self.grid = np.zeros(self.bins)
        ''' initialize 2d and 1d bins'''
        self.avg_bins_2d = int(self.avg_bins / 2)
        self.ranges_2d = []
        for i in range(3):
            self.ranges_2d.append((min(self.data_list[i]), max(self.data_list[i])))
        self.bin_space_2d = np.mean([i[1] - i[0] for i in self.ranges_2d]) / self.avg_bins_2d
        self.bins_2d = [math.ceil((self.ranges_2d[i][1] - self.ranges_2d[i][0]) / self.bin_space_2d) for i in range(2)]
        self.bin_space_1d = (np.max(self.data_list[2]) - np.min(self.data_list[2])) / self.avg_bins_2d
        self.bins_1d = self.avg_bins_2d
        self.grid_2d = np.zeros(self.bins_2d)
        self.grid_1d = np.zeros(self.bins_1d)
        print('binspace: ', self.bin_space)
        print('bins: ', self.bins)
        print('ranges: ', self.ranges)
        print('2d and 1d bins shape: ', self.grid_2d.shape, self.grid_1d.shape)
        print('2d and 1d bins space: ', self.bin_space_2d, self.bin_space_1d)

    def __repr__(self):
        return 'sky<bin space: {:f}, avg bins: {:f}>'.format(self.bin_space, self.avg_bins)

    def _coord_to_grid(self, point):
        if len(point) < 3:
            raise ValueError('Point (list) has to have at least length 3')
        if isinstance(point[0], np.ndarray):
            index = []
            for i in range(3):
                v = (point[i] - self.ranges[i][0]) / self.bin_space
                v = v.astype(int)
                v[v >= self.bins[i]] = -1
                index.append(v)
            index = np.asarray(index)
            index[index <= 0] = -1
            index = index.astype(int)
            return index
        index = np.array(
            [min(self.bins[i] - 1, int((point[i] - self.ranges[i][0]) / self.bin_space)) for i in range(3)])
        index[index <= 0] = -1
        return index

    def _coord_to_grid_2d(self, point):
        if len(point) < 3:
            raise ValueError('Point (list) has to have at least length 3')
        if isinstance(point[0], np.ndarray):
            index = []
            for i in range(2):
                v = (point[i] - self.ranges_2d[i][0]) / self.bin_space_2d
                v[v >= self.bins_2d[i]] = -1
                index.append(v)
            z = (point[2] - self.ranges_2d[2][0]) / self.bin_space_1d
            z[z >= self.bins_1d] = -1
            index.append(z)
            index = np.asarray(index)
            index[index <= 0] = -1
            index = index.astype(int)
            return index

        if np.isnan(point).any():
            # return [0, 0, 0]
            return [-1, -1, -1]
        # print(point[2], self.ranges_2d[2][0], self.bin_space_1d)

        index = [min(self.bins_2d[i] - 1, int((point[i] - self.ranges_2d[i][0]) / self.bin_space_2d)) for i in range(2)]
        z = int((point[2] - self.ranges_2d[2][0]) / self.bin_space_1d)
        if z >= self.bins_1d:
            z = -1
        index.append(z)
        index = np.asarray(index)
        index[index <= 0] = -1
        return index

    def _grid_to_coord(self, point):
        if len(point) != 3:
            raise ValueError('Grid coordinate has to have dimension 3; get {:d}'.format(len(point)))
        if isinstance(point[0], np.ndarray):
            index = []
            for i in range(3):
                v = point[i] * self.bin_space + self.ranges[i][0]
                # v = v.astype(int)
                v[v >= self.ranges[i][1]] = np.nan
                index.append(v)
            index = np.asarray(index)
            return index
        print(point)
        index = np.array(
            [min(self.ranges[i][1], point[i] * self.bin_space + self.ranges[i][0]) for i in range(3)])
        return index

    def get_center_grid(self):
        center_grid = np.zeros(self.grid.shape)
        centers = self.get_centers(grid=True).T
        print(centers)
        center_grid[centers[0], centers[1], centers[2]] = 1
        return center_grid

    def vote2(self, radius):
        grids = self._coord_to_grid(self.xyz_list)
        self.grid[grids[0], grids[1], grids[2]] += 1
        w = util.sphere_window(radius, self.bin_space)
        self.grid = util.conv(self.grid, w)

    def vote(self, radius, error=-1):
        if error == -1:
            error = self.bin_space
        for point in zip(*self.xyz_list):
            if np.isnan(point).any():
                pass
            else:
                sphere = util.draw_sphere(point, radius, self.bin_space, error)
                sphere_2d = [util.cartesian_to_sky(point) for point in sphere]
                sphere_2d = np.array(
                    [self._coord_to_grid_2d(p) for p in sphere_2d if self._coord_to_grid_2d(p) is not None]).T
                sphere_2d = sphere_2d[:, ~np.any(sphere_2d == -1, axis=0)]
                sphere = np.array([self._coord_to_grid(p) for p in sphere]).T
                sphere = sphere[:, ~np.any(sphere == -1, axis=0)]

                self.grid[sphere[0], sphere[1], sphere[2]] += 1
                self.grid_1d[sphere_2d[2]] += 1
                self.grid_2d[sphere_2d[0], sphere_2d[1]] += 1
        #   Normalizing 2d and 1d bins
        self.grid_2d /= self.grid_2d.sum()
        self.grid_1d /= self.grid_1d.sum()
        print('binning finshished----------------------')

    def blobs_thres(self, radius, blob_size, type_, error=-1):
        self.grid[0, :, :] = 0
        self.grid[:, 0, :] = 0
        self.grid[:, :, 0] = 0
        print(self.bins)
        if error == -1:
            error = self.bin_space
        threshold = self.get_threshold(radius, type_=type_)
        blobs = dog(self.grid, threshold, type_, blob_size)
        print('blob finshished----------------------')
        print('blobs: ', blobs)
        blobs = [self._grid_to_coord(p) for p in blobs]
        print('blobs: ', blobs)

        ret = np.array(np.where(self.grid >= threshold)).T
        print('number above threshold: ', len(ret))
        self.centers = np.asarray(blobs)
        self.centers = self.fit_bao(radius, error * 2)

        return blobs

    def eval(self):
        if len(self.centers) == 0:
            return [], 0
        print(self.centers)
        centers_d = self.get_centers()
        centers_d = np.asarray(centers_d)
        print('centers len', centers_d)
        distribution = []
        centers_found = 0
        for center in self.centers:
            dist_2 = np.sum((centers_d[:, :3] - center[:3]) ** 2, axis=1)
            nearest_dist = np.min(dist_2) ** .5
            distribution.append(nearest_dist)
        print(np.vstack(self.centers).shape)
        for center_d in centers_d:
            dist_2 = np.sum((center_d[:3] - np.vstack(self.centers)) ** 2, axis=1)
            nearest_dist = np.min(dist_2) ** .5
            print(nearest_dist)
            if nearest_dist < 18:
                centers_found += 1
        print('c found: ', centers_found)
        return distribution, centers_found

    def draw_sphere(self, point, radius, error=-1):
        if error == -1:
            error = self.bin_space
        diameter_bins = int((radius + error) * 2 / self.bin_space)
        x, y, z = point[:3]
        sphere_coord_x = np.linspace(x - radius - error, x + radius, diameter_bins + error)
        sphere_coord_y = np.linspace(y - radius - error, y + radius, diameter_bins + error)
        sphere_coord_z = np.linspace(z - radius - error, z + radius, diameter_bins + error)
        a, b, c = np.meshgrid(sphere_coord_x, sphere_coord_y, sphere_coord_z)
        sphere = zip(a.ravel(), b.ravel(), c.ravel())
        sphere = [point2 for point2 in sphere if radius - error < util.distance(point, point2) < radius + error]
        sphere = np.array([self._coord_to_grid(p) for p in sphere]).T
        self.grid[sphere[0], sphere[1], sphere[2]] += 1
        return sphere

    def place_sphere(self, point, window, radius, error):
        if any(x % 1 != 0 for x in point):
            raise ValueError('Point coordinates has to be converted to indices first')
        dist = int((window.shape[0] - 1) / 2)
        # print(window.shape)
        # print(self.grid[point[0] - dist: point[0] + dist + 1, point[1] - dist: point[1] + dist + 1,
        # point[2] - dist: point[2] + dist + 1].shape)
        # print('--------------------')
        # self.grid[point[0] - dist: point[0] + dist + 1, point[1] - dist: point[1] + dist + 1,
        # point[2] - dist: point[2] + dist + 1] += window
        try:
            self.grid[point[0] - dist: point[0] + dist + 1, point[1] - dist: point[1] + dist + 1,
            point[2] - dist: point[2] + dist + 1] += window
        except:
            self.draw_sphere(point, radius, error)
        return 0

    def get_centers(self, grid=False):
        ret = np.vstack([point for point in self.points_arr if point[3] == 2])
        if grid:
            return np.asarray(self._coord_to_grid(ret.T)).T
        else:
            return ret

    def fit_bao(self, radius, error):
        ret = []
        xyz = np.array(self.xyz_list)[:3].T
        print('Fitting bao')
        for center in self.centers:
            sphere = [p for p in xyz if
                      (util.distance(center, p) <= radius + error) and (util.distance(center, p) >= radius - error)]
            sphere = np.array(sphere).T
            if len(sphere) == 0:
                pass
            else:
                # print('points found in sphere: ', sphere.shape)
                fit = util.sphere_fit(sphere[0], sphere[1], sphere[2])
                # print(fit)
                ret.append(fit[1])
        try:
            print('fit: ', type(np.vstack(ret)))
            return np.vstack(ret)
        except:
            return np.zeros((3, 0))

    def plot_sky(self, show_rim=False, radius=0, savelabel=None) -> None:
        ax = Axes3D(plt.gcf())
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        centers_d = self.get_centers()
        ax.scatter(centers_d[:, 0], centers_d[:, 1], centers_d[:, 2], color='blue')
        for center in self.centers:
            ax.scatter(center[0], center[1], center[2], color='red')
        print('centers: ', len(self.centers))
        plt.title(savelabel)
        if show_rim:
            if not radius:
                raise ValueError('Need radius')
            ax.scatter(xs=self.xyz_list[0], ys=self.xyz_list[1], zs=self.xyz_list[2], color='blue', alpha=0.1)
            for fit in self.centers:
                sph = util.draw_sphere(fit, radius, self.bin_space)
                sph = np.array(sph).T
                if len(sph) != 0:
                    ax.scatter(xs=sph[0], ys=sph[1], zs=sph[2], alpha=0.05, color='r')
            ax.legend(['rim in data', 'center in data', 'rim found', 'center found'])
        else:
            ax.legend(['center in data', 'center found'])

        if savelabel:
            path = '../Figures/Figure_' + savelabel + '.png'
            plt.savefig(path)
        else:
            plt.show()

    def plot_original(self) -> None:
        ax = Axes3D(plt.gcf())
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        centers_d = self.get_centers()
        ax.scatter(centers_d[:, 0], centers_d[:, 1], centers_d[:, 2], color='blue')
        ax.scatter(xs=self.xyz_list[0], ys=self.xyz_list[1], zs=self.xyz_list[2], color='blue', alpha=0.1)
        plt.show()

    def plot_eval(self, savelabel: str=None) -> None:
        rc('font', family='serif')
        # rc('font', size=16)
        distr, center = self.eval()
        mean, std = norm.fit(distr)
        num = len(np.where((distr >= mean - 3 * std) & (distr <= mean + 3 * std)))
        ratio = num / len(distr)
        efficiency = len([x for x in distr if x < 18]) / center
        fake = 1 - efficiency
        plt.figure(figsize=[6, 6])
        plt.hist(distr, bins=50, density=True)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        y = norm.pdf(x, mean, std)
        plt.plot(x, y)

        plt.xlabel(
            'distance \n centers found = {:d}\n mean = {:f}, standard deviation = {:f}\nefficiency = {:f}, fake rate = {:f}'.format(
                len(distr), mean, std, efficiency, fake))
        plt.ylabel('frequency')
        plt.title('Distance to real centers (SignalN3_mid)')

        plt.tight_layout()
        if savelabel:
            path = '../Figures/Figure_' + savelabel + '_distr.png'
            plt.savefig(path)
        else:
            plt.show()

    def get_threshold(self, radius, type_='ratio'):

        '''
        dVang=R^2 dR/dredshift dredshift cos(de) dde dalpha

        dR/dredshift=norm(1-3redshift*Omega_M/2)

        dVD=dx dy dz

        weight=dVD/dVang

        norm = 3000,

        Omega_M=0.274,

        de - declination, dde - step in declination

        alpha - right ascension, dalpha - step in  right ascension

        dredshift - step in redshift

        Nexp=Ntot*Pang*Pz*weight.

        '''
        # self.grid -= util.local_thres(self.grid, 20)
        #return util.local_thres(self.grid, 20)
        norm = 3000
        Omega_M = 0.274
        dVD = self.bin_space ** 3
        x, y, z = [np.arange(self.bins[i]) for i in range(3)]
        thres_grid = np.vstack(np.meshgrid(x, y, z, indexing='ij')).reshape(3, -1)
        thres_grid = self._grid_to_coord(thres_grid)
        ra, dec, z = util.cartesian_to_sky(thres_grid)
        drdz = norm * (1 - 3 * z * Omega_M / 2)
        dVang = radius ** 2 * drdz * self.bin_space_1d * np.cos(np.radians(dec)) * self.bin_space_2d ** 2
        weight = dVD / dVang
        new_idx = self._coord_to_grid_2d([ra, dec, z])
        thres_grid = self.grid_2d[new_idx[0], new_idx[1]] * self.grid_1d[new_idx[2]] * weight
        thres_grid = thres_grid.reshape(self.grid.shape)
        if type_ == 'ratio':
            median = np.median(thres_grid)
            thres_grid[thres_grid < median] = np.nan

            factor = abs(np.median(self.grid) / median)
            print('factor: ', factor)
            thres_grid *= factor
        elif type_ == 'difference':
            print('thres_grid: ', thres_grid)
            median = np.median(thres_grid)
            thres_grid[thres_grid < median] = np.nan
            #factor = abs(np.median(self.grid) / median)
            factor = abs(np.nanmean(self.grid) / np.nanmean(thres_grid))

            print('factor: ', factor)
            thres_grid *= factor
        # section = int(len(thres_grid) / 4)
        # plt.imshow(self.grid[section] * 2)
        # plt.show()
        # plt.imshow(thres_grid[section * 2])
        # plt.show()
        # plt.imshow(thres_grid[section * 3])
        # plt.show()

        # f, axarr = plt.subplots(3, 3)
        # new = self.grid / thres_grid
        # axarr[0, 0].imshow(self.grid[:, :, section])
        # axarr[0, 1].imshow(thres_grid[:, :, section])
        # axarr[0, 2].imshow(new[:, :, section])
        # axarr[1, 0].imshow(self.grid[:, :, section * 2])
        # axarr[1, 1].imshow(thres_grid[:, :, section * 2])
        # axarr[1, 2].imshow(new[:, :, section * 2])
        # axarr[2, 0].imshow(self.grid[:, :, section * 3])
        # axarr[2, 1].imshow(thres_grid[:, :, section * 3])
        # axarr[2, 2].imshow(new[:, :, section * 3])
        # cols = ['N-obs', 'N-exp', 'N-obs/N-exp']
        # for ax, col in zip(axarr[0], cols):
        #     ax.set_title(col)
        # plt.tight_layout()
        # plt.show()
        # plt.savefig('../Figures/Figure_sampling_{:d}.png'.format(radius))

        return thres_grid
