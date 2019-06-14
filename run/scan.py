import numpy as np
from astropy.io import fits
import pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os

from centerfinder import util
from centerfinder import blob
from centerfinder import sky



def read_fits(filename):
	if filename.split('.')[-1] == 'fits':
		hdul = fits.open(filename)
		ra = hdul[1].data['RA']
		dec = hdul[1].data['DEC']
		z = hdul[1].data['Z']
		typ_ = hdul[1].data['WEIGHT']  # placeholder
		return np.vstack([ra, dec, z, typ_])


def vote(filename):
	for radius in range(90, 130, 3):
		sky_ = sky.Sky(util.load_data(filename), 5)
		sky_.vote(radius=radius)
		sky_.vote_2d_1d()
		path = filename.split('.')[-2].split('/')[-1]
		path = '../output/' + path + '_' + str(radius)
		print(path, filename)
		util.pickle_sky(sky_, path)


def blob(filename):

	for radius in range(90, 130, 3):
		
		# get the voted result
		file = filename + '_' + str(radius)
		sky_ = util.unpickle_sky(file)

		# blob
		sky_.blobs_thres(radius = radius, blob_size = 5, type_ = 'difference', rms_factor = 1)
		filename_new = '../output/{}'.format(filename) + file.split('/')[-1] + '_blob'
		util.pickle_sky(sky_, filename_new)


def test_blob(sky_, radius, rms_factor=1, filename=None):

	sky_.blobs_thres(radius=radius, blob_size=5, type_='difference', rms_factor=rms_factor)
	util.pickle_sky(sky_, filename)
	center_num = len(sky_.centers)
	ev = sky_.eval()
	eff = ev[1]
	center_f_true = ev[2]
	with open(filename+ '_rms.txt', 'w') as f:
		f.write(str(center_num))
		f.write('\n')
		f.write(str(rms_factor))
		f.write('\n')
		f.write(str(eff))
		f.write('\n')
		f.write(str(center_f_true))
		f.write('\n---------------------------------------\n')
	return [radius, rms_factor, center_num, eff, center_f_true]


def scan_rms(filename):
	lst = []
	sky_ = util.unpickle_sky(filename)
	if not isinstance(sky_, sky.Sky):
		raise ValueError("Object is of type " + type(sky_))
	for rms in np.arange(0.9, 2, 0.1):
		print(rms)
		tmp = test_blob(sky_, 108, rms_factor=rms, filename=filename)
		lst.append(tmp)
	lst = np.asarray(lst)
	with open(filename + '_rms_pickle', 'wb') as f:
		pickle.dump(lst, f)


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
		sky_ = util.unpickle_sky(file)

		all_centers_num = len(sky_.centers)
		total_centers.append(all_centers_num)
		total_err.append(np.sqrt(all_centers_num))

		# evaluation true centers
		ev = sky_.eval()
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
	args = parser.parse_args()

	# making the specific output directory inside the general output directory
	filename = args.file.split('.')[-2]
	os.mkdir('../output/%s' % filename)

	if (args.vote or args.full):
		vote('../data/%s' % args.file)

	if (args.blob or args.full):
		blob('../output/%s/%s' % (filename, filename))

	if (args.eval or args.full):
		eval(filename, args.nr_true_centers)

	if (args.plot or args.full):
		dataset = ' '.join(filename.split('_')[1:5])
		plot(dataset, filename)



