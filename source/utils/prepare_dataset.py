#!/usr/bin/python
import os
import json
import glob
import random
import skimage
import argparse
import threading
import numpy as np
from tqdm import tqdm
from math import ceil
from itertools import chain
from utils import cut_string
import matplotlib.pyplot as plt


def worker(n, files, result, classes, colors):
	pbar = tqdm(files, position=n)
	for file in pbar:
		pbar.set_description(cut_string(file))
		dir, name = os.path.dirname(file), os.path.splitext(os.path.basename(file))[0]
		mpath = os.path.join(dir, name + '.mask.png')
		mask = skimage.io.imread(mpath)
		mask = np.apply_along_axis(lambda x: x == colors, 2, mask).sum(axis=3) == 3
		ind = [i for i in range(len(classes)) if mask[::, ::, i].sum() > 0]
		bpath = os.path.join(dir, name)
		np.savez_compressed(bpath, mask=mask[::, ::, ind], classes=classes[ind])
		result[n].append({ 'image': file, 'mask': mpath, 'boolmask': bpath + '.npz'})


parser = argparse.ArgumentParser(description='Prepare dataset')
parser.add_argument('directory', metavar='DIR', help='Path to dataset', type=str)
parser.add_argument('--train', metavar='P', help='Train weight', type=float, default=0.8)
parser.add_argument('--classes', metavar='C', help='Path to classes json', default='types.json')
parser.add_argument('-t', '--threads', type=int, default=1, help='Number of threads', metavar='N')
args = parser.parse_args()

assert args.train <= 1, 'Train weight must be in range [0, 1]'

classes = np.array(json.load(open(os.path.join(args.directory, args.classes))))
colors = np.array([(i, i, i, 1) for i in range(1, len(classes) + 1)], dtype='uint8')
files = list(filter(lambda x: not x.endswith('.mask.png'), glob.iglob(os.path.join(args.directory, '**/*.png'), recursive=True)))
random.shuffle(files)

result, threads, tn = [[] for i in range(args.threads)], [], ceil(len(files) / args.threads)

for i in range(args.threads):
	threads.append(threading.Thread(target=worker, args=(i, files[i * tn:(i + 1) * tn], result, classes, colors)))
	threads[-1].start()

for i in threads:
	i.join()

for i in threads:
	print()

k = int(len(files) * args.train)
result = list(chain(*result))
with open('./train.json', 'w') as train, open('./test.json', 'w') as test:
	json.dump(result[:k:], train, indent=4)
	json.dump(result[k::], test, indent=4)
