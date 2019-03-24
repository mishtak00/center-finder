from centerfinder import util
from centerfinder import sky
from centerfinder import eval
from argparse import ArgumentParser


def plot_chi_sqr():
    parser = ArgumentParser(description="Center finder")
    parser.add_argument('filename', metavar='F', type=str, nargs=1,
                        help='Path to data file')
    parser.add_argument('radius', metavar='R', type=int, nargs=1,
                        help='Expected radius')
    args = parser.parse_args()
    filename = args.filename[0]
    radius = args.radius[0]
    sky_ = sky.Sky(util.load_data(filename), 5)
    sky_.find_center(radius=radius, blob_size=3, type_='difference')
    eval.plot_chi_sqr(sky_, radius=radius)


def chi_sqr_vs_radius():
    parser = ArgumentParser(description="Center finder")
    parser.add_argument('filename', metavar='F', type=str, nargs=1,
                        help='Path to data file')
    args = parser.parse_args()
    filename = args.filename[0]
    eval.plot_over_radius(filename, 0, eval.avg_chi_sqr, label='average chi-square over radius')


if __name__ == '__main__':
    chi_sqr_vs_radius()
