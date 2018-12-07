from numpy import ones, triu, seterr
from numpy.linalg import norm
from math import pi
from scipy.ndimage.filters import gaussian_laplace, minimum_filter


def localMinima(data, threshold):
    from numpy import ones, nonzero, transpose

    if threshold is not None:
        peaks = data < threshold
    else:
        peaks = ones(data.shape, dtype=data.dtype)

    peaks &= data == minimum_filter(data, size=(3,) * data.ndim)
    return transpose(nonzero(peaks))


def blobLOG(data, scales=range(1, 10, 1), threshold=-30):
    """Find blobs. Returns [[scale, x, y, ...], ...]"""
    from numpy import empty, asarray

    data = asarray(data)
    scales = asarray(scales)

    log = empty((len(scales),) + data.shape, dtype=data.dtype)
    for slog, scale in zip(log, scales):
        slog[...] = scale ** 2 * gaussian_laplace(data, scale)

    peaks = localMinima(log, threshold=threshold)
    peaks[:, 0] = scales[peaks[:, 0]]
    return peaks


def sphereIntersection(r1, r2, d):
    # https://en.wikipedia.org/wiki/Spherical_cap#Application

    valid = (d < (r1 + r2)) & (d > 0)
    return (pi * (r1 + r2 - d) ** 2
            * (d ** 2 + 6 * r2 * r1
               + 2 * d * (r1 + r2)
               - 3 * (r1 - r2) ** 2)
            / (12 * d)) * valid


def circleIntersection(r1, r2, d):
    # http://mathworld.wolfram.com/Circle-CircleIntersection.html
    from numpy import arccos, sqrt

    return (r1 ** 2 * arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
            + r2 ** 2 * arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
            - sqrt((-d + r1 + r2) * (d + r1 - r2)
                   * (d - r1 + r2) * (d + r1 + r2)) / 2)


def findBlobs(img, scales=range(1, 10), threshold=30, max_overlap=0.05):
    old_errs = seterr(invalid='ignore')

    peaks = blobLOG(img, scales=scales, threshold=-threshold)
    radii = peaks[:, 0]
    positions = peaks[:, 1:]

    distances = norm(positions[:, None, :] - positions[None, :, :], axis=2)

    if positions.shape[1] == 2:
        intersections = circleIntersection(radii, radii.T, distances)
        volumes = pi * radii ** 2
    elif positions.shape[1] == 3:
        intersections = sphereIntersection(radii, radii.T, distances)
        volumes = 4 / 3 * pi * radii ** 3
    else:
        raise ValueError("Invalid dimensions for position ({}), need 2 or 3."
                         .format(positions.shape[1]))

    delete = ((intersections > (volumes * max_overlap))
              # Remove the smaller of the blobs
              & ((radii[:, None] < radii[None, :])
                 # Tie-break
                 | ((radii[:, None] == radii[None, :])
                    & triu(ones((len(peaks), len(peaks)), dtype='bool'))))
              ).any(axis=1)

    seterr(**old_errs)
    return peaks[~delete]


def peakEnclosed(peaks, shape, size=1):
    from numpy import asarray
    shape = asarray(shape)
    return (size <= peaks).all(axis=-1) & (size < (shape - peaks)).all(axis=-1)
