import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from blob import findBlobs
from scipy.stats import norm
import util


class Sky:
    def __init__(self, xyz_list, avg_bins=100):
        if len(xyz_list) != 0:
            self.xyz_list = xyz_list
            self.points_arr = np.vstack(self.xyz_list).T

        else:
            raise ValueError('List of coordinates has wrong shape')
        self.ranges = []
        for i in range(3):
            self.ranges.append((min(xyz_list[i]), max(xyz_list[i])))
        self.bin_space = np.mean([i[1] - i[0] for i in self.ranges]) / avg_bins
        print(self.bin_space)
        self.bins = [math.ceil((self.ranges[i][1] - self.ranges[i][0]) / self.bin_space) for i in range(3)]
        print(self.bins)
        self.grid = np.zeros(self.bins)

    def __repr__(self):
        return 'sky<\n\tranges: \n\t\t' + str(self.ranges) +'\n\tbin space:\n\t\t' + str(self.bin_space) + '\n>'

    def coord_to_grid(self, point):
        index = np.array(
            [min(self.bins[i] - 1, int((point[i] - self.ranges[i][0]) / self.bin_space)) for i in range(3)])
        index[index < 0] = 0
        return index

    def grid_to_coord(self, point):
        index = np.array(
            [min(self.ranges[i][1], point[i] * self.bin_space + self.ranges[i][0]) for i in range(3)])
        return index

    def center_finder(self, radius, error=-1, center_num=10, threshold=10):
        if error == -1:
            error = self.bin_space
        for point in zip(*self.xyz_list):
            if np.isnan(point).any():
                pass
            else:
                sphere = util.draw_sphere(point, radius, self.bin_space, error)
                sphere = np.array([self.coord_to_grid(p) for p in sphere]).T
                self.grid[sphere[0], sphere[1], sphere[2]] += 1

        #
        blobs = findBlobs(self.grid, scales=range(1, 2), threshold=threshold)
        blobs = np.array(blobs)
        print('shape: ', blobs.shape)

        blobs = [self.grid_to_coord(p[1:]) for p in blobs]
        print('blobs: \n', blobs)

        ret = np.array(np.where(self.grid >= threshold)).T
        print(len(ret))
        self.centers = np.asarray(blobs)
        return blobs

    def eval(self):
        if not self.centers.any():
            raise ValueError("self.centers not defined. Run center_finder routine first")
        centers_d = self.get_centers()
        centers_d = np.asarray(centers_d)
        print(centers_d)
        print(self.centers)
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

    def fit_bao(self, radius):
        ret = []
        xyz = np.array(self.xyz_list)[:3].T
        print(xyz.shape)
        for center in self.centers:
            print('center: ', center)
            sphere = [p for p in xyz if (util.distance(center, p) < radius * 1.2)]
            sphere = np.array(sphere).T
            print(sphere.shape)
            fit = util.sphere_fit(sphere[0], sphere[1], sphere[2])
            print(fit)
            ret.append(fit)
        return ret

    def plot_sky(self, show_rim=False, radius=0):
        ax = Axes3D(plt.gcf())
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        centers_d = self.get_centers()
        ax.scatter(centers_d[:, 0], centers_d[:, 1], centers_d[:, 2], color='blue')
        for center in self.centers:
            ax.scatter(center[0], center[1], center[2], color='red')
        if show_rim:
            if not radius:
                raise ValueError('Need radius')
            ax.scatter(xs=self.xyz_list[0], ys=self.xyz_list[1], zs=self.xyz_list[2], color='blue', alpha=0.1)
            for fit in self.centers:
                sph = util.draw_sphere(fit, radius, self.bin_space)
                sph = np.array(sph).T
                ax.scatter(xs=sph[0], ys=sph[1], zs=sph[2], alpha=0.05, color='r')
            ax.legend(['rim in data', 'center in data', 'rim found', 'center found'])
        else:
            ax.legend(['center in data', 'center found'])
        plt.show()

    def plot_eval(self):
        distr = self.eval()
        mean, std = norm.fit(distr)
        print(mean, std)
        plt.hist(distr, bins=20, density=True)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        y = norm.pdf(x, mean, std)
        plt.plot(x, y)
        plt.show()
