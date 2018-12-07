import numpy as np
import math
from blob import findBlobs
import util


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
        print(self.bin_space)
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

    def merge_bins(self, factor):
        self.bin_space *= factor
        dim = [math.floor(col / factor) for col in self.grid.shape]
        self.bins = dim
        ret = self.grid[:factor * dim[0], :factor * dim[1], :factor * dim[2]].reshape(dim[0], factor, dim[1], factor,
                                                                                      dim[2], factor)
        ret = ret.max(axis=(1, 3, 5))
        return ret

    def center_finder(self, radius, error=-1, center_num=10):
        if error == -1:
            error = self.bin_space
        for point in zip(*self.xyz_list):
            sphere = util.draw_sphere(point, radius, self.bin_space, error)
            sphere = np.array([self.coord_to_grid(p) for p in sphere]).T
            self.grid[sphere[0], sphere[1], sphere[2]] += 1

        sorted_grid = sorted(self.grid.flatten())
        threshold = sorted_grid[-center_num]
        blobs = findBlobs(self.grid, scales=range(1, 3), threshold=10)
        blobs = np.array(blobs)
        print('shape: ', blobs.shape)

        blobs = [self.grid_to_coord(p[1:]) for p in blobs]
        print('blobs: \n', blobs)

        ret = np.array(np.where(self.grid >= threshold)).T
        print(len(ret))
        self.centers = blobs
        return blobs, threshold

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
