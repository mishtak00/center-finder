import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from .blob import dog
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
from typing import List
from . import util


class Sky:
    def __init__(self, data_list: np.ndarray, space_3d) -> None:
        """  
        :param data_list: from files
        :param space_3d: 3-d bin space
        """
        if len(data_list) >= 3:
            self.data_list = data_list
        else:
            raise ValueError('data list has wrong dimension')
        self.centers = []
        self.xyz_list = util.sky_to_cartesian(data_list)

        ''' initialize params'''
        self.range_3d = []
        self.range_2d = []
        self.range_1d = []
        for i in range(3):
            self.range_3d.append((min(self.xyz_list[i]), max(self.xyz_list[i])))
        for i in range(2):
            self.range_2d.append((min(self.data_list[i]), max(self.data_list[i])))
        self.range_1d = (min(self.data_list[2]), max(self.data_list[2]))

        self.space_3d = space_3d
        self.bins_3d = [math.ceil((self.range_3d[i][1] - self.range_3d[i][0]) / self.space_3d) for i in range(3)]

        self.avg_bins = int(np.mean(self.bins_3d))

        self.space_2d = np.mean([i[1] - i[0] for i in self.range_2d]) / self.avg_bins / 2
        self.bins_2d = [math.ceil((self.range_2d[i][1] - self.range_2d[i][0]) / self.space_2d) for i in range(2)]
        self.space_1d = (self.range_1d[1] - self.range_1d[0]) / self.avg_bins / 2
        self.bins_1d = self.avg_bins * 2

        '''initialize bins'''
        self.grid = np.zeros(self.bins_3d)
        self.grid_2d = np.zeros(self.bins_2d)
        self.grid_1d = np.zeros(self.bins_1d)

        self.centers = []
        print(self, file=sys.stderr)

    def __repr__(self):
        return '***************** sky parameters *****************\n' \
               '3d range: {}\n' \
               'dec ra range: {}\n' \
               'z range: {}\n' \
               '3d bin space: {}'.format(
            self.range_3d, self.range_2d,
            self.range_1d, self.space_3d)

    def _coord_to_grid(self, point):
        if len(point) < 3:
            raise ValueError('Point (list) has to have at least length 3')
        if isinstance(point[0], np.ndarray):
            index = []
            for i in range(3):
                v = (point[i] - self.range_3d[i][0]) / self.space_3d
                v = v.astype(int)
                v[v >= self.bins_3d[i]] = -1
                index.append(v)
            index = np.asarray(index)
            index[index <= 0] = -1
            index = index.astype(int)
            return index
        index = np.array(
            [min(self.bins_3d[i] - 1, int((point[i] - self.range_3d[i][0]) / self.space_3d)) for i in range(3)])
        index[index <= 0] = -1
        return index

    def _coord_to_grid_2d(self, point):
        if len(point) < 3:
            raise ValueError('Point (list) has to have at least length 3')
        if isinstance(point[0], np.ndarray):
            index = []
            for i in range(2):
                v = (point[i] - self.range_2d[i][0]) / self.space_2d
                v[v >= self.bins_2d[i]] = -1
                index.append(v)
            z = (point[2] - self.range_1d[0]) / self.space_1d
            z[z >= self.bins_1d] = -1
            index.append(z)
            index = np.asarray(index)
            index[index <= 0] = -1
            index = index.astype(int)
            return index

        if np.isnan(point).any():
            return [-1, -1, -1]
        index = [min(self.bins_2d[i] - 1, int((point[i] - self.range_2d[i][0])
                                              / self.space_2d)) for i in range(2)]
        z = int((point[2] - self.range_1d[0]) / self.space_1d)
        if z >= self.bins_1d:
            z = -1
        index.append(z)
        index = np.asarray(index)
        index[index <= 0] = -1
        return index.astype(int)

    def _grid_to_coord(self, point):
        if len(point) != 3:
            raise ValueError('Grid coordinate has to have dimension 3; get {:d}'.format(len(point)))
        if isinstance(point[0], np.ndarray):
            index = []
            for i in range(3):
                v = point[i] * self.space_3d + self.range_3d[i][0]
                v[v >= self.range_3d[i][1]] = np.nan
                index.append(v)
            index = np.asarray(index)
            return index
        index = np.array(
            [min(self.range_3d[i][1], point[i] * self.space_3d + self.range_3d[i][0]) for i in range(3)])
        return index

    def get_center_grid(self):
        """
        Create a 3-d grid with only true centers in it
        :return: ndarray (shape = self.grid.shape)
        """
        center_grid = np.zeros(self.grid.shape)
        centers = self.get_centers(grid=True).T
        center_grid[centers[0], centers[1], centers[2]] = 1
        return center_grid

    def get_found_center_grid(self):
        """
        Create a 3-d grid with only found centers in it
        :return: ndarray (shape = self.grid.shape)
        """
        center_grid = np.zeros(self.grid.shape)
        centers = self.centers.T
        centers = self._coord_to_grid(centers)
        center_grid[centers[0], centers[1], centers[2]] = 1
        return center_grid

    def get_galaxy_grid(self):
        """
        Create a 3-d grid with all galaxies in it
        :return: ndarray (shape = self.grid.shape)
        """
        galaxy_grid = np.zeros(self.grid.shape)
        grids = self._coord_to_grid(self.xyz_list)
        galaxy_grid[grids[0], grids[1], grids[2]] = 1
        return galaxy_grid

    def find_center(self, radius: [float, int],
                    blob_size: int,
                    type_: str,
                    error=-1):
        """
        The wrapper function of center-finding procedure
        Run it and get the centers
        :param radius: expected BAO radius
        :param blob_size: blob size
        :param type_: 'difference' or 'ratio'
        :param error:
        :return:
        """
        self.vote(radius)
        self.blobs_thres(radius, blob_size, type_, error)
        self.confirm_center()

    def vote(self, radius: float) -> None:
        # TODO: fix 2d and 1d binning
        """
        Core function; does the voting procedure as a 3-D convolution
        :param radius: BAO radius
        :return: None
        """
        grids = self._coord_to_grid(self.xyz_list)
        self.grid[grids[0], grids[1], grids[2]] += 1
        w = util.kernel(radius, self.space_3d)
        self.grid = util.conv(self.grid, w)

    def blobs_thres(self,
                    radius: [int, float],
                    blob_size: int,
                    type_: str,
                    rms_factor=1.5,
                    error=-1) -> None:
        """
        Find blobs on voted grid
        :param radius: expected BAO radius
        :param blob_size:
        :param type_: ratio or difference (for threshold-ing uses)
        :param rms_factor: customize factor in blob finding; for testing use
        :param error: error allowed in fit_bao
        :return: None
        """
        self.grid[0, :, :] = 0
        self.grid[:, 0, :] = 0
        self.grid[:, :, 0] = 0
        if error == -1:
            error = self.space_3d
        # threshold = self.get_threshold(radius)
        threshold = util.local_thres(self.grid, 40)
        blobs = dog(self.grid, threshold, type_, blob_size, rms_factor)
        sys.stderr.write('***************** blob finished *****************\n')
        self.centers = np.asarray(blobs)
        sys.stderr.write('number of centers found (before confirming): {}\n'.format(len(blobs)))
        #
        # filter out found centers > 18 Mpcs away from any galaxy
        gaus = util.conv(self.get_galaxy_grid(), util.distance_kernel(18, self.space_3d))
        confirmed = []
        print(len(blobs))
        for c_grid in blobs:
            c_grid = [int(c) for c in c_grid]
            if gaus[c_grid[0], c_grid[1], c_grid[2]] > 0:
                confirmed.append(self._grid_to_coord(c_grid))
        self.centers = np.asarray(confirmed)
        print(self.centers.shape)
        self.centers = self.fit_bao(radius, error * 2)
        self.centers = np.asarray(self.centers)
        sys.stderr.write('number of centers found: {}\n'.format(len(confirmed)))

    def confirm_center(self) -> None:
        """
        Remove found centers that are > 18 Mpcs away from any galaxy
        :return: None
        """
        if len(self.centers) == 0:
            sys.stderr.write('No centers found')
        else:
            gaus = util.conv(self.get_galaxy_grid(), util.distance_kernel(18, self.space_3d))
            confirmed = []
            for c_grid in self.centers:
                c_grid = [int(c) for c in self._coord_to_grid(c_grid)]
                if gaus[c_grid[0], c_grid[1], c_grid[2]] > 0:
                    confirmed.append(self._grid_to_coord(c_grid))
            self.centers = np.asarray(confirmed)
            sys.stderr.write('number of centers found: {}\n'.format(len(confirmed)))

    def eval(self):
        """
        Evaluate center-finding results
        :return:
            distribution: of distances between each found center and its nearest true center
            center_true: number of true centers within 18 Mpcs to nearest found center
        """
        if len(self.centers) == 0:
            return [], 0
        centers_d = self.get_centers()
        centers_d = np.asarray(centers_d)
        distribution = []
        centers_true = 0

        # calculate centers_true
        center_f_grid = self.get_found_center_grid()
        center_f_grid = util.conv(center_f_grid, util.distance_kernel(18, self.space_3d))
        for center in self.get_centers(grid=True):
            if center_f_grid[center[0], center[1], center[2]] > 0:
                centers_true += 1

        # calculate distribution
        for center in self.centers:
            dist_2 = np.sum((centers_d[:, :3] - center[:3]) ** 2, axis=1)
            nearest_dist = np.min(dist_2) ** .5
            distribution.append(nearest_dist)
        return distribution, centers_true

    def draw_sphere(self, point, radius, error=-1):
        """
        For plotting purpose
        :param point:
        :param radius:
        :param error:
        :return:
        """
        if error == -1:
            error = self.space_3d
        diameter_bins = int((radius + error) * 2 / self.space_3d)
        x, y, z = point[:3]
        sphere_coord_x = np.linspace(x - radius - error, x + radius, diameter_bins + error)
        sphere_coord_y = np.linspace(y - radius - error, y + radius, diameter_bins + error)
        sphere_coord_z = np.linspace(z - radius - error, z + radius, diameter_bins + error)
        a, b, c = np.meshgrid(sphere_coord_x, sphere_coord_y, sphere_coord_z)
        sphere = zip(a.ravel(), b.ravel(), c.ravel())
        sphere = [point2 for point2 in sphere if
                  radius - error < util.distance(point, point2) < radius + error]
        sphere = np.array([self._coord_to_grid(p) for p in sphere]).T
        self.grid[sphere[0], sphere[1], sphere[2]] += 1
        return sphere

    def get_centers(self, grid=False):
        ret = np.vstack([point for point in self.xyz_list.T if point[3] == 2])
        if grid:
            return np.asarray(self._coord_to_grid(ret.T)).T
        else:
            return ret

    def fit_bao(self, radius: [int, float], error: float) -> np.ndarray:
        ret = []
        xyz = np.array(self.xyz_list)[:3].T
        for center in self.centers:
            sphere = [p for p in xyz if
                      (util.distance(center, p) <= radius + error) and
                      (util.distance(center, p) >= radius - error)]
            sphere = np.array(sphere).T
            if len(sphere) == 0:
                pass
            else:
                fit = util.sphere_fit(sphere[0], sphere[1], sphere[2])
                ret.append(fit[1])
        try:
            return np.vstack(ret)
        except:
            return np.zeros((3, 0))

    def plot_sky(self, show_rim=False, radius=0, savelabel=None) -> None:
        """
        Plots the true centers and found centers
        :param show_rim: show the rim galaxies
        :param radius:
        :param savelabel:
            if provided, save the image with savelabel
            else simply show the image without saving
        :return:
        """
        ax = Axes3D(plt.gcf())
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        centers_d = self.get_centers()
        ax.scatter(centers_d[:, 0], centers_d[:, 1], centers_d[:, 2], color='blue')
        for center in self.centers:
            ax.scatter(center[0], center[1], center[2], color='red')
        plt.title(savelabel)
        if show_rim:
            if not radius:
                raise ValueError('Need radius')
            ax.scatter(xs=self.xyz_list[0],
                       ys=self.xyz_list[1],
                       zs=self.xyz_list[2],
                       color='blue', alpha=0.1)
            for fit in self.centers:
                sph = util.draw_sphere(fit, radius, self.space_3d)
                sph = np.array(sph).T
                if len(sph) != 0:
                    ax.scatter(xs=sph[0], ys=sph[1], zs=sph[2], alpha=0.05, color='r')
            ax.legend(['rim in data', 'center in data', 'rim found', 'center found'])
        else:
            ax.legend(['center in data', 'center found'])

        if savelabel:
            path = 'Figures/Figure_' + savelabel + '.png'
            plt.savefig(path)
        else:
            plt.show()

    def plot_original(self) -> None:
        """
        Plots the given catalog: centers in red, rims in blue
        :return: None
        """
        ax = Axes3D(plt.gcf())
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        centers_d = self.get_centers()
        ax.scatter(centers_d[:, 0], centers_d[:, 1], centers_d[:, 2], color='blue')
        ax.scatter(xs=self.xyz_list[0],
                   ys=self.xyz_list[1],
                   zs=self.xyz_list[2],
                   color='blue', alpha=0.1)
        plt.show()

    def plot_eval(self, centers_generated: int, savelabel: str = None) -> None:
        """
        Plots the center-finding result:
        distribution of distances between found centers to nearest true centers
        :param centers_generated:
        :param savelabel:
            if provided, save the image with savelabel
            else show the image without saving
        :return: None
        """
        rc('font', family='serif')
        # rc('font', size=16)
        distr, true_centers = self.eval()
        mean, std = norm.fit(distr)
        found_eff = len([d for d in distr if d <= 18])
        efficiency = found_eff / centers_generated
        fake_rate = (len(self.centers) - found_eff) / len(self.centers)
        multiplicity = found_eff / true_centers

        plt.figure(figsize=[6, 6])
        plt.hist(distr, bins=50, density=True)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 50)
        y = norm.pdf(x, mean, std)
        plt.plot(x, y)

        plt.xlabel(
            'distance \n'
            'centers found = {:d} '
            'mean = {:f}, standard deviation = {:f}\n'
            'efficiency = {:f}, fake rate = {:f}, multiplicity = {:f}'.format(
                len(distr), mean, std, efficiency, fake_rate, multiplicity))
        plt.ylabel('frequency')
        plt.title('Distance to real centers (SignalN3_mid)')

        plt.tight_layout()
        if savelabel:
            path = 'Figures/Figure_' + savelabel + '_distr.png'
            plt.savefig(path)
        else:
            plt.show()

    def get_threshold(self, radius):
        """
        Threshold from ra-dec and z bins (2d and 1d bins)
        :param radius:
        :return: threshold (same shape as the 3-d bin)
        """
        # calculation from formula
        norm = 3000
        Omega_M = 0.274
        dVD = self.space_3d ** 3
        x, y, z = [np.arange(self.bins_3d[i]) for i in range(3)]
        thres_grid = np.vstack(np.meshgrid(x, y, z, indexing='ij')).reshape(3, -1)
        thres_grid = self._grid_to_coord(thres_grid)
        ra, dec, z = util.cartesian_to_sky(thres_grid)
        drdz = norm * (1 - 3 * z * Omega_M / 2)
        dVang = radius ** 2 * drdz * self.space_1d * np.cos(np.radians(dec)) * self.space_2d ** 2
        weight = dVD / dVang

        # combine 2d and 1d grids into 3d grid
        new_idx = self._coord_to_grid_2d([ra, dec, z])
        thres_grid = self.grid_2d[new_idx[0], new_idx[1]] * self.grid_1d[new_idx[2]] * weight
        thres_grid = thres_grid.reshape(self.grid.shape)

        # scale threshold
        median = np.median(thres_grid)
        thres_grid[thres_grid < median] = np.nan
        factor = abs(np.median(self.grid) / median)
        #thres_grid *= factor
        return thres_grid

    def get_voters(self, center, radius, abs_idx=False):
        """
        Get all the voters for a given center, returned in grid indices
        Default: return relative indices to the given center
        :param center:
        :param radius: expected BAO radius
        :param abs_idx: set abs_idx to true to get the real indices of voters
        :return:
        """
        x, y, z = self._coord_to_grid(center)
        r_bins = int(radius / self.space_3d)
        r_bins += 1

        # get the sub-array centered at 'center'
        subgrid = self.get_galaxy_grid()[x-r_bins:x+r_bins, y-r_bins:y+r_bins, z-r_bins:z+r_bins]
        rim_list = []
        center = r_bins
        r_bins -= 1
        for idx, val in np.ndenumerate(subgrid):
            dist = util.distance(idx, [center, center, center])
            if r_bins - 1 < dist < r_bins + 1 and val > 0:
                rim_list.append(idx)
            else:
                subgrid[idx] = 0
        if abs_idx:
            rim_list = [(r[0]+x, r[1]+y, r[2]+z) for r in rim_list]

        return np.asarray(rim_list)