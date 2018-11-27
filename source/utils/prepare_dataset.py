#!/usr/bin/python
import os
import json
import glob
import time
import random
import tables
import skimage
import argparse
import warnings
import threading
import numpy as np
from tqdm import tqdm
from math import ceil
from queue import Queue
from utils import cut_string
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class Wrapper:

	def __init__(self, value):
		self.value = value

def worker(n, files, result, classes, colors, count, lock, queue):
	pbar = tqdm(files, position=n)
	for file in pbar:
		pbar.set_description(cut_string(file))
		dir, name = os.path.dirname(file), os.path.splitext(os.path.basename(file))[0]
		mask = skimage.io.imread(os.path.join(dir, name + '.mask.png'))
		mask = np.apply_along_axis(lambda x: x == colors, 2, mask).sum(axis=3) == 3
		ind = [i for i in range(len(classes)) if mask[::, ::, i].sum() > 0]
		lock.acquire()
		queue.put_nowait(('image' + str(count), skimage.io.imread(file)[:, :, :3]))
		queue.put_nowait(('mask' + str(count), mask[:, :, ind]))
		queue.put_nowait(('classes' + str(count), classes[ind]))
		result.append(count)
		lock.release()
		count += 1


def writer(filename, queue, is_done, lock):
	file = tables.open_file(filename, mode='w')
	filters = tables.Filters(complevel=5, complib='blosc')
	while True:
		time.sleep(1)
		lock.acquire()
		b = queue.empty()
		if is_done.value and b:
			return
		while not b:
			name, value = queue.get_nowait()
			lock.release()
			arr = file.create_carray(file.root, name, tables.Atom.from_dtype(value.dtype), shape=value.shape, filters=filters)
			arr[:] = value
			lock.acquire()
			b = queue.empty()
		lock.release()


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

result, tn, queue, lock, is_done, threads = [], ceil(len(files) / args.threads), Queue(), threading.Lock(), Wrapper(False), []

writer_thread = threading.Thread(target=writer, args=('dataset.hdf5', queue, is_done, lock))
writer_thread.start()

for i in range(args.threads):
	threads.append(threading.Thread(target=worker, args=(i, files[i * tn:(i + 1) * tn], result, classes, colors, i * tn, lock, queue)))
	threads[-1].start()

for i in threads:
	i.join()

for i in threads:
	print()

lock.acquire()
is_done.value = True
lock.release()

print('Writing files')
writer_thread.join()

k = int(len(files) * args.train)
with open('./train.json', 'w') as train, open('./test.json', 'w') as test:
	json.dump(result[:k:], train)
	json.dump(result[k::], test)
