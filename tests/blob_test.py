from centerfinder import util
from centerfinder import sky
from argparse import ArgumentParser


if __name__ == '__main__':

    parser = ArgumentParser(description="Center finder")
    parser.add_argument('filename', metavar='F', type=str, nargs=1,
                        help='Path to data file')
    parser.add_argument('centers', metavar='C', type=int, nargs=1,
                        help='Number of generated centers in mock')
    args = parser.parse_args()
    filename = args.filename[0]
    centers = args.centers[0]
    sky_ = sky.Sky(util.load_data(filename), 5)
    # expected radius should be default to 108
    sky_.vote(radius=108)
    sky_.blobs_thres(radius=108, blob_size=3, type_='difference')
    sky_.plot_sky()
    # need to know centers_generated
    sky_.plot_eval(centers_generated=centers)