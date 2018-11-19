`#!/usr/bin/python
import argparse
import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
from itertools import chain

parser = argparse.ArgumentParser(description='Stuffs')
parser.add_argument('folder', help='Folder', type=str)
parser.add_argument('-b', '--bins', help='Bins', type=int, default=10)
args = parser.parse_args()

def count_ratio(x):
	return np.count_nonzero(x[::, ::, 3], axis=0).sum() / (x.shape[0] * x.shape[1])

folder = os.path.join('./', args.folder)

for d in filter(lambda x: os.path.isdir(x), map(lambda x: os.path.join(folder, x), os.listdir(folder))):
	fig, ax = plt.subplots()
	ax.hist(np.fromiter((count_ratio(mpimg.imread(file)) for file in filter(lambda x: os.path.isfile(x), map(lambda x: os.path.join(d, x), os.listdir(d)))), dtype=float), bins=args.bins, color=np.random.rand(3))
	ax.set_title(d)
plt.show()
