from centerfinder import util
from centerfinder import sky
from centerfinder import eval
from argparse import ArgumentParser


def plot_func():
    parser = ArgumentParser(description="Center finder")
    parser.add_argument('filename', metavar='F', type=str, nargs=1,
                        help='Path to data file')
    parser.add_argument('centers', metavar='C', type=int, nargs=1,
                        help='Number of generated centers in mock')
    args = parser.parse_args()
    filename = args.filename[0]
    centers = args.centers[0]

    # To customize the function to plot, change func={eval.fake_rate, eval.centers_found}
    eval.plot_over_radius(filename=filename,
                          center_num=centers,
                          func=eval.centers_found,
                          bin_space=5,
                          radius_step=3)


if __name__ == '__main__':
    plot_func()