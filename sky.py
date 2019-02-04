import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from blob import findBlobs, simple_maxima
from scipy.stats import norm
import util


class Sky:
    def __init__(self, data_list, avg_bins=100, bin_space=0):
        if len(data_list) > 0:
            self.data_list = data_list
        else:
            raise ValueError('Data list cannot be empty')
        self.xyz_list = util.SkyToSphere(*data_list)
        self.points_arr = np.vstack(self.xyz_list).T

        ''' initialize 3d bins'''
        self.ranges = []
        for i in range(3):
            self.ranges.append((min(self.xyz_list[i]), max(self.xyz_list[i])))
        self.bin_space = np.mean([i[1] - i[0] for i in self.ranges]) / avg_bins

        if bin_space:
            self.bin_space = bin_space

        print('binspace: ', self.bin_space)
        self.bins = [math.ceil((self.ranges[i][1] - self.ranges[i][0]) / self.bin_space) for i in range(3)]
        avg_bins = np.mean(self.bins)
        print('bins: ', self.bins)
        print('ranges: ', self.ranges)
        self.grid = np.zeros(self.bins)

        ''' initialize 2d and 1d bins'''
        avg_bins_2d = int(avg_bins / 2)
        self.ranges_2d = []
        for i in range(3):
            self.ranges_2d.append((min(self.data_list[i]), max(self.data_list[i])))
        self.bin_space_2d = np.mean([i[1] - i[0] for i in self.ranges_2d]) / avg_bins_2d
        self.bins_2d = [math.ceil((self.ranges_2d[i][1] - self.ranges_2d[i][0]) / self.bin_space_2d) for i in range(2)]
        self.bin_space_1d = (np.max(self.data_list[2]) - np.min(self.data_list[2])) / avg_bins_2d
        self.bins_1d = avg_bins_2d
        self.grid_2d = np.zeros(self.bins_2d)
        self.grid_1d = np.zeros(self.bins_1d)
        print('2d and 1d bins shape: ', self.grid_2d.shape, self.grid_1d.shape)
        print('2d and 1d bins space: ', self.bin_space_2d, self.bin_space_1d)

    def __repr__(self):
        return 'sky<\n\tranges: \n\t\t' + str(self.ranges) + '\n\tbin space:\n\t\t' + str(self.bin_space) + '\n>'

    def coord_to_grid(self, point):
        index = np.array(
            [min(self.bins[i] - 1, int((point[i] - self.ranges[i][0]) / self.bin_space)) for i in range(3)])
        index[index < 0] = 0
        return index

    def coord_to_grid_2d(self, point):
        # print('2d ranges: ', self.ranges_2d)
        # print('bin spaces: ', self.bin_space_2d, self.bin_space_1d)

        if np.isnan(point).any():
            # return [0, 0, 0]
            return None
        # print(point[2], self.ranges_2d[2][0], self.bin_space_1d)
        index = [min(self.bins_2d[i] - 1, int((point[i] - self.ranges_2d[i][0]) / self.bin_space_2d)) for i in range(2)]
        z = min(self.bins_1d - 1, int((point[2] - self.ranges_2d[2][0]) / self.bin_space_1d))
        index.append(z)
        index = np.asarray(index)
        index[index < 0] = 0
        return index

    def grid_to_coord(self, point):
        index = np.array(
            [min(self.ranges[i][1], point[i] * self.bin_space + self.ranges[i][0]) for i in range(3)])
        return index

    def center_finder(self, radius, error=-1, blob_size=10, threshold=None):
        if error == -1:
            error = self.bin_space
        for point in zip(*self.xyz_list):
            if np.isnan(point).any():
                pass
            else:
                sphere = util.draw_sphere(point, radius, self.bin_space, error)
                sphere_2d = [util.CartesianToSky(*point) for point in sphere]
                sphere_2d = np.array(
                    [self.coord_to_grid_2d(p) for p in sphere_2d if self.coord_to_grid_2d(p) is not None]).T
                # sphere_2d = [p for p in sphere if p is not None]
                sphere = np.array([self.coord_to_grid(p) for p in sphere]).T

                self.grid[sphere[0], sphere[1], sphere[2]] += 1
                self.grid_1d[sphere_2d[2]] += 1
                self.grid_2d[sphere_2d[0], sphere_2d[1]] += 1

        print('binning finshished----------------------')
        self.grid_2d /= self.grid_2d.sum()
        self.grid_1d /= self.grid_1d.sum()
        if not threshold:
            threshold = self.get_threshold(radius)
        print('threshold finished----------------------')
        #self.grid -= threshold
        self.grid /= threshold
        self.grid -= util.local_thres(self.grid, 20) / 2

        # blob_idx = np.where(self.grid >= threshold/2)
        #
        # blobs = np.vstack(blob_idx).T
        # blobs = [self.grid_to_coord(b) for b in blobs]

        blobs = findBlobs(self.grid, scales=range(1, blob_size), threshold=100)
        blobs = np.array(blobs)
        print('blobs: ', blobs.shape)

        blobs = [self.grid_to_coord(p[1:]) for p in blobs]
        print('blobs: \n', blobs)

        ret = np.array(np.where(self.grid >= threshold)).T
        print('number above threshold: ', len(ret))
        print(self.grid_2d)
        self.centers = np.asarray(blobs)
        # self.centers = self.fit_bao(radius, error)
        return blobs

    def blobs(self, threshold, radius, error, blob_size):

        # blob_idx = np.where(self.grid >= 300)
        # print(blob_idx)
        # blobs = np.vstack(blob_idx).T
        # blobs = [self.grid_to_coord(b) for b in blobs]

        blobs = findBlobs(self.grid, scales=range(1, blob_size), threshold=15)
        print(blobs)
        print('blobs: ', blobs.shape)
        blobs = [self.grid_to_coord(p[1:]) for p in blobs]
        print('blobs: \n', blobs)

        ret = np.array(np.where(self.grid >= threshold)).T
        print('number above threshold: ', len(ret))
        self.centers = np.asarray(blobs)
        self.centers = self.fit_bao(radius, error)

        return blobs

    def blobs_thres(self, threshold, radius, error, blob_size):

        self.grid /= threshold
        self.grid -= util.local_thres(self.grid, 20)/2
        # blob_idx = np.where(self.grid >= 150)
        # print(blob_idx)
        # blobs = np.vstack(blob_idx).T
        # blobs = [self.grid_to_coord(b) for b in blobs]

        blobs = findBlobs(self.grid, scales=range(1, blob_size), threshold=5)
        print(blobs)
        print('blobs: ', blobs.shape)
        blobs = [self.grid_to_coord(p[1:]) for p in blobs]
        print('blobs: \n', blobs)

        # blobs = simple_maxima(self.grid)
        # print(blobs)
        # print('blobs: ', blobs.shape)
        # blobs = [self.grid_to_coord(p) for p in blobs]
        # print('blobs: \n', blobs)

        ret = np.array(np.where(self.grid >= threshold)).T
        print('number above threshold: ', len(ret))
        self.centers = np.asarray(blobs)
        self.centers = self.fit_bao(radius, error * 2)

        return blobs

    def eval(self):
        print(self.centers)
        if not self.centers:
            raise ValueError("self.centers not defined. Run center_finder routine first")
        centers_d = self.get_centers()
        centers_d = np.asarray(centers_d)
        print('centers len', len(self.centers))
        distribution = []
        for center in self.centers:
            print('self.center: ', center[:3])
            dist_2 = np.sum((centers_d[:, :3] - center[:3]) ** 2, axis=1)
            nearest_dist = np.min(dist_2) ** .5
            print('dist: ', nearest_dist)
            distribution.append(nearest_dist)
        return distribution

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
        sphere = np.array([self.coord_to_grid(p) for p in sphere]).T
        self.grid[sphere[0], sphere[1], sphere[2]] += 1
        return sphere

    def place_sphere(self, point, window, radius, error):
        if any(x % 1 != 0 for x in point):
            raise ValueError('Point coordinates has to be converted to indices first')
        dist = int((window.shape[0] - 1) / 2)
        try:
            self.grid[point[0] - dist: point[0] + dist + 1, point[1] - dist: point[1] + dist + 1,
            point[2] - dist: point[2] + dist + 1] += window
        except:
            self.draw_sphere(point, radius, error)
        return 0

    def get_centers(self):
        return np.vstack([point for point in self.points_arr if point[3] == 2])

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
        return np.vstack(ret)

    def plot_sky(self, show_rim=False, radius=0, savelabel=None):
        ax = Axes3D(plt.gcf())
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        centers_d = self.get_centers()
        ax.scatter(centers_d[:, 0], centers_d[:, 1], centers_d[:, 2], color='blue')
        for center in self.centers:
            ax.scatter(center[0], center[1], center[2], color='red')
        print('centers: ', len(self.centers))
        plt.title('SignalN3_mid')
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
            path = 'Figures/Figure_' + savelabel + '.png'
            plt.savefig(path)
        else:
            plt.show()

    def plot_original(self):
        ax = Axes3D(plt.gcf())
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        centers_d = self.get_centers()
        ax.scatter(centers_d[:, 0], centers_d[:, 1], centers_d[:, 2], color='blue')
        ax.scatter(xs=self.xyz_list[0], ys=self.xyz_list[1], zs=self.xyz_list[2], color='blue', alpha=0.1)
        plt.show()

    def plot_eval(self, savelabel=None):
        rc('font', family='serif')
        # rc('font', size=16)
        distr = self.eval()
        mean, std = norm.fit(distr)
        num = len(np.where((distr >= mean - 3 * std) & (distr <= mean + 3 * std)))
        ratio = num / len(distr)
        efficiency = len([x for x in distr if x < 20]) / len(distr)
        fake = 1 - efficiency
        print(mean, std, ratio)
        print('\n')
        print("Ratio within 3 sigma of mean: ", ratio)
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
        # plt.figtext(0.5, 0.02, "mean = {:f}, standard deviation = {:f}\nefficiency = {:f}".format(mean, std, efficiency), wrap=True, horizontalalignment='center', fontsize=12)

        plt.tight_layout()
        if savelabel:
            path = 'Figures/Figure_' + savelabel + '_distr.png'
            plt.savefig(path)
        else:
            plt.show()

    def get_threshold(self, radius):
        # if not (self.grid and self.grid_2d and self.grid_1d):
        #     raise ValueError('Run center-finding routine first')
        '''

        :return:
        '''

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

        print(self.grid_2d)
        norm = 3000
        Omega_M = 0.274
        dVD = self.bin_space ** 3
        x, y, z = self.grid.shape
        thres_grid = np.dstack(np.meshgrid(np.arange(x), np.arange(y), indexing='ij'))
        # print(thres_grid)
        thres_grid = np.zeros(self.grid.shape)
        for idx, val in np.ndenumerate(thres_grid):
            sky_coord = self.grid_to_coord(idx)
            # print('sky coord: ', sky_coord)
            ra, dec, z = util.CartesianToSky(*(sky_coord[:3]))

            drdz = norm * (1 - 3 * z * Omega_M / 2)
            dVang = radius ** 2 * drdz * self.bin_space_1d * np.cos(np.radians(dec)) * self.bin_space_2d ** 2
            weight = dVD / dVang
            new_idx = self.coord_to_grid_2d([ra, dec, z])
            if new_idx is not None:
                thres_grid[idx] = self.grid_2d[new_idx[0], new_idx[1]] * self.grid_1d[new_idx[2]] * weight

        factor = abs(np.median(self.grid) / np.median(thres_grid))
        print('factor: ', factor)
        print(np.mean(self.grid), np.median(self.grid), np.max(self.grid))
        thres_grid *= factor
        print(thres_grid)
        # thres_grid += util.local_thres(self.grid, 20)/2
        return thres_grid

    def get_hard_thres(self):
        '''TODO: not working'''
        sorted = np.sort(self.grid.flatten())
        rank = int(19 * len(sorted) / 20)
        thres = sorted[rank]
        print('hard threshold: ', thres)
        return thres
