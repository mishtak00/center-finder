import numpy as np
from astropy.io import fits
import pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os

from centerfinder import util
from centerfinder import sky as s


def read_fits(filename):
    if filename.split('.')[-1] == 'fits':
        hdul = fits.open(filename)
        ra = hdul[1].data['RA']
        dec = hdul[1].data['DEC']
        z = hdul[1].data['Z']
        n_observed = hdul[1].data['N_OBSERVED']
        n_expected = hdul[1].data['N_EXPECTED']
        return np.vstack([ra, dec, z, n_observed, n_expected])


def write_fits(filename, centers, n_observed, n_expected):
    columns = []
    columns.append(fits.Column(name='RA', format='E', array=centers[0]))
    columns.append(fits.Column(name='DEC', format='E', array=centers[1]))
    columns.append(fits.Column(name='Z', format='E', array=centers[2]))
    columns.append(fits.Column(name='N_OBSERVED', format='E', array=n_observed))
    columns.append(fits.Column(name='N_EXPECTED', format='E', array=n_expected))
    new_hdus = fits.BinTableHDU.from_columns(columns)
    try:
        new_hdus.writeto('{}.fits'.format(filename))
    except OSError:
        pass
    print('\nThe following is the output in .fits:\n', read_fits('{}.fits'.format(filename)))


def vote(filename):
    for radius in range(90, 130, 3):
        sky = s.Sky(util.load_data(filename), 5)
        sky.vote(radius=radius)
        sky.vote_2d_1d()
        path = filename.split('.')[-2].split('/')[-1]
        path = '../output/{}/'.format(path) + path + '_' + str(radius)
        util.pickle_sky(sky, path)

        # output to .fits
        galaxies = np.asarray(sky.data_list, dtype=np.float64)
        n_observed = np.asarray(sky.observed_grid.flatten(), dtype=int)
        columns = []
        columns.append(fits.Column(name='RA', format='E', array=galaxies[0]))
        columns.append(fits.Column(name='DEC', format='E', array=galaxies[1]))
        columns.append(fits.Column(name='Z', format='E', array=galaxies[2]))
        columns.append(fits.Column(name='N_OBSERVED', format='E', array=n_observed))
        new_hdus = fits.BinTableHDU.from_columns(columns)
        path += '.fits'
        try:
            new_hdus.writeto(path)
        except OSError:
            os.remove(path)
            new_hdus.writeto(path)


def blob(filename):

    for radius in range(90, 130, 3):

        # get the voted result
        file = filename + '_' + str(radius)
        sky = util.unpickle_sky(file)

        # blob
        sky.blobs_thres(radius=radius, blob_size=5, type_='difference', rms_factor=1)

        # TODO: FIX THIS DATA FORMAT PROBLEM FOR Y'S AND Z'S
        centers = np.asarray(sky.centers, dtype=np.float64)

        # This gets the voters per each center found
        n_observed = np.asarray(sky.observed, dtype=int)

        # This gets the expected voters per each center found
        n_expected = np.asarray(sky.expected, dtype=np.float64)

        # This puts data in celestial coordinates
        centers = util.cartesian_to_sky(centers.T)
        print('Centers\' shape for fits output:', centers.shape)

        filename_new = file + '_blob'

        # Output to .pkl
        # util.pickle_sky(sky, filename_new)

        # Output to .fits
        write_fits(filename_new, centers, n_observed, n_expected)


def test_vote(filename, radius=108):

    sky = s.Sky(util.load_data(filename), 5)
    sky.vote(radius=radius)
    sky.vote_2d_1d()
    path = filename.split('.')[-2].split('/')[-1]
    path = '../output/{}/'.format(path) + path + '_' + str(radius)
    util.pickle_sky(sky, path)

    # output to .fits
    galaxies = np.asarray(sky.data_list, dtype=np.float64)
    n_observed = np.asarray(sky.observed_grid.flatten(), dtype=int)
    columns = []
    columns.append(fits.Column(name='RA', format='E', array=galaxies[0]))
    columns.append(fits.Column(name='DEC', format='E', array=galaxies[1]))
    columns.append(fits.Column(name='Z', format='E', array=galaxies[2]))
    columns.append(fits.Column(name='N_OBSERVED', format='E', array=n_observed))
    new_hdus = fits.BinTableHDU.from_columns(columns)
    path += '.fits'
    try:
        new_hdus.writeto(path)
    except OSError:
        os.remove(path)
        new_hdus.writeto(path)


def test_blob(filename, radius=108):

    # get the voted result
    file = filename + '_' + str(radius)
    sky = util.unpickle_sky(file)

    # blob
    sky.blobs_thres(radius=radius, blob_size=5, type_='difference', rms_factor=1)

    # TODO: FIX THIS DATA FORMAT PROBLEM FOR Y'S AND Z'S
    centers = np.asarray(sky.centers, dtype=np.float64)

    # This gets the voters per each center found
    n_observed = np.asarray(sky.centers_n_observed, dtype=int)

    # This gets the expected voters per each center found
    n_expected = np.asarray(sky.centers_n_expected, dtype=np.float64)

    # This puts data in celestial coordinates
    centers = util.cartesian_to_sky(centers.T)
    print('Centers\' shape for fits output:', centers.shape)

    filename_new = file + '_blob'

    # Output to .pkl
    # util.pickle_sky(sky, filename_new)

    # Output to .fits
    write_fits(filename_new, centers, n_observed, n_expected)

    # Output n_exp vs n_obs
    plt.title('N_observed vs. N_expected')
    plt.xlabel('N_obs')
    plt.ylabel('N_exp')
    plt.scatter(n_observed, n_expected, s=7)
    plt.savefig(file + '.png')


# def test_blob(sky, radius, rms_factor=1, filename=None):
#   sky.blobs_thres(radius=radius, blob_size=5, type_='difference', rms_factor=rms_factor)
#   util.pickle_sky(sky, filename)
#   center_num = len(sky.centers)
#   ev = sky.eval()
#   eff = ev[1]
#   center_f_true = ev[2]
#   with open(filename+ '_rms.txt', 'w') as f:
#       f.write(str(center_num))
#       f.write('\n')
#       f.write(str(rms_factor))
#       f.write('\n')
#       f.write(str(eff))
#       f.write('\n')
#       f.write(str(center_f_true))
#       f.write('\n---------------------------------------\n')
#   return [radius, rms_factor, center_num, eff, center_f_true]


# def scan_rms(filename):
#   lst = []
#   sky = util.unpickle_sky(filename)
#   if not isinstance(sky, s.Sky):
#       raise ValueError("Object is of type " + type(sky))
#   for rms in np.arange(0.9, 2, 0.1):
#       print(rms)
#       tmp = test_blob(sky, 108, rms_factor=rms, filename=filename)
#       lst.append(tmp)
#   lst = np.asarray(lst)
#   with open(filename + '_rms_pickle', 'wb') as f:
#       pickle.dump(lst, f)


def eval(filename, centers):

    fake_list = []
    fake_error = []

    efficiency_list = []
    eff_error = []

    total_centers = []
    total_err = []

    true_centers = []
    true_err = []

    # not for the clumping mocks
    nr_true_centers = centers

    for radius in range(90, 130, 3):

        # get the blob result
        file = '../output/{}'.format(filename) + filename + '_' + str(radius) + '_blob'
        sky = util.unpickle_sky(file)

        all_centers_num = len(sky.centers)
        total_centers.append(all_centers_num)
        total_err.append(np.sqrt(all_centers_num))

        # evaluation true centers
        ev = s.eval()
        center_f_true = ev[2]

        true_centers.append(center_f_true)
        true_err.append(np.sqrt(center_f_true))

        efficiency_list.append((center_f_true / nr_true_centers))
        eff_e = np.sqrt(center_f_true) / nr_true_centers
        eff_error.append(eff_e)

        num_fake = all_centers_num - center_f_true
        fake_list.append((num_fake / all_centers_num))
        fake_e = np.sqrt(num_fake) / all_centers_num
        fake_error.append(fake_e)

    all_big_list = [fake_list, fake_error, efficiency_list, eff_error, total_centers, total_err, true_centers, true_err]
    with open('../output/{}/eval_output'.format(filename), 'wb') as f:
        pickle.dump(all_big_list, f)


def plot(dataset, filename):

    with open('../output/eval_output', 'rb') as f:
        all_big_list = pickle.load(f)

    fake_list = all_big_list[0]
    fake_error = all_big_list[1]

    efficiency_list = all_big_list[2]
    eff_error = all_big_list[3]

    total_centers = all_big_list[4]
    total_err = all_big_list[5]

    true_centers = all_big_list[6]
    true_err = all_big_list[7]

    rs = range(90, 130, 3)

    plt.figure(1)
    plt.errorbar(rs, efficiency_list, eff_error, capsize=8, ls='', marker='o', ms=8, mfc='darkblue', mec='k', label=r'{}'.format(dataset), ecolor='k')
    plt.xlabel('Radius')
    plt.ylabel('Efficiency')
    plt.title('%s Efficiency (RMS = 1)' % dataset)
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig('../output/%s Efficiency (RMS = 1).png' % dataset)

    plt.figure(2)
    plt.errorbar(rs, fake_list, fake_error, capsize=8, ls='', marker='o', ms=8, mfc='darkblue', mec='k', label=r'{}'.format(dataset), ecolor='k')
    plt.xlabel('Radius')
    plt.ylabel('Fake Rate')
    plt.title('%s Fake Rate (RMS = 1)' % dataset)
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig('../output/%s Fake Rate (RMS = 1).png' % dataset)

    plt.figure(3)
    plt.errorbar(rs, total_centers, total_err, capsize=8, ls='', marker='o', ms=8, mfc='darkblue', mec='k', label=r'{}'.format(dataset), ecolor='k')
    plt.xlabel('Radius')
    plt.ylabel('Total Number of Centers')
    plt.title('%s Total Number of Centers (RMS = 1)' % dataset)
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig('../output/%s Total Number of Centers (RMS = 1).png' % dataset)

    plt.figure(4)
    plt.errorbar(rs, true_centers, true_err, capsize=8, ls='', marker='o', ms=8, mfc='darkblue', mec='k', label=r'{}'.format(dataset), ecolor='k')
    plt.xlabel('Radius')
    plt.ylabel('Number of True Centers Found')
    plt.title('%s Number of True Centers Found (RMS = 1)' % dataset)
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig('../output/%s Number of True Centers Found (RMS = 1).png' % dataset)


if __name__ == '__main__':
    parser = ArgumentParser(description='( * ) Center finder ( * )')
    parser.add_argument('file', metavar='FILE', type=str, help='Name of fits file to be fitted.')
    parser.add_argument('nr_true_centers', metavar="NR_TRUE_CENTERS", type=int,
                        help="Number of true centers in the dataset to be analyzed. \
            This is needed for the evaluation step.")
    parser.add_argument('--vote', action='store_true', help='If this argument is present, the "vote" procedure will occur.')
    parser.add_argument('--blob', action='store_true', help='If this argument is present, the "blob" procedure will occur.')
    parser.add_argument('--eval', action='store_true', help='If this argument is present, the "eval" procedure will occur.')
    parser.add_argument('--plot', action='store_true', help='If this argument is present, the "plot" procedure will occur.')
    parser.add_argument('--full', action='store_true', help='If this argument is present, all of the procedures will occur.')
    parser.add_argument('--test_blob', action='store_true', help='If this argument is present, the blob testing procedures will occur.')
    parser.add_argument('--test_vote', action='store_true', help='If this argument is present, the vote testing procedures will occur.')
    args = parser.parse_args()

    # making the specific output directory inside the general output directory
    filename = args.file.split('.')[-2]
    try:
        os.mkdir('../output/%s' % filename)
    except:
        pass

    if (args.vote or args.full):
        vote('../data/%s' % args.file)

    if (args.blob or args.full):
        blob('../output/%s/%s' % (filename, filename))

    if (args.test_blob):
        test_blob('../output/%s/%s' % (filename, filename))

    if (args.test_vote):
        test_vote('../data/%s' % args.file)

    if (args.eval or args.full):
        eval(filename, args.nr_true_centers)

    if (args.plot or args.full):
        # this gets 1 thru 5 of the filename split because it assumes
        # the filename is of the form cf_set_i_mock_j_etc.etc
        dataset = ' '.join(filename.split('_')[1:5])
        plot(dataset, filename)
