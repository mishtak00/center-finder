import os.path as osp
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import nbodykit

def load_data(filename):
    # filename = osp.join(osp.abspath('../data'), filename)
    with open(filename) as f:
        for line in f:
            tuple = line.split()