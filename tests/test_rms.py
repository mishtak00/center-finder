import numpy as np
import matplotlib.pyplot as plt
from centerfinder import util
from centerfinder import blob
from centerfinder import sky as sk
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
from centerfinder import eval

filename = '../data/cf_mock_catalog_333C_30R_randoms_added.fits'
sky_ = sk.Sky(util.load_data(filename), 5)
