#!/usr/bin/python
import os
import json
import glob
import time
import random
import tables
import skimage
import argparse
import threading
import numpy as np
from tqdm import tqdm
from math import ceil
from queue import Queue
from utils import cut_string
import matplotlib.pyplot as plt


class Wrapper:

	def __init__(self, value):
		self.value = value


def unique(uclasses, classes):
	new, uc = [], uclasses[:]
	for i in classes:
		if i not in uc and i not in new:
			new.append(i)
	if new:
		new.sort()
		uclasses.append(new)


def worker(n, files, result, classes, uclasses, colors, count, lock, queue):
	pbar = tqdm(files, position=n)
	uclasses = ['background'] + list(set(classes))
	for file in pbar:
		pbar.set_description(cut_string(file))
		dir, name = os.path.dirname(file), os.path.splitext(os.path.basename(file))[0]
		mask = skimage.io.imread(os.path.join(dir, name + '.mask.png'))
		mask = np.apply_along_axis(lambda x: x == colors, 2, mask).sum(axis=3) == 3
		ind = [i for i in range(len(classes)) if mask[::, ::, i].sum() > 0]
		lock.acquire()
		queue.put_nowait(('image', count, skimage.io.imread(file)[:, :, :3]))
		queue.put_nowait(('mask', count, mask[:, :, ind]))
		queue.put_nowait(('class_id', count, np.array([uclasses.index(i) + 1 for i in classes[ind]], dtype='uint8')))
		result.append(count)
		lock.release()
		count += 1


def writer(file, filters, queue, is_done, lock):
	count = file.root.count[0]
	while True:
		time.sleep(1)
		lock.acquire()
		b = queue.empty()
		if is_done.value and b:
			return
		while not b:
			category, id, value = queue.get_nowait()
			id += count
			lock.release()
			arr = file.create_carray(file.root[category], '_' + str(id), obj=value, filters=filters)
			lock.acquire()
			b = queue.empty()
		lock.release()


parser = argparse.ArgumentParser(description='Prepare dataset')
parser.add_argument('directory', metavar='DIR', help='Path to dataset', type=str)
parser.add_argument('-f', '--file', type=str, default='dataset.hdf5', help='HDF5 dataset file', metavar='F')
parser.add_argument('--train', metavar='P', help='Train weight', type=float, default=0.8)
parser.add_argument('--classes', metavar='C', help='Path to classes json', default='types.json')
parser.add_argument('-t', '--threads', type=int, default=1, help='Number of threads', metavar='N')
parser.add_argument('--complevel', type=int, default=9, help='Compression level', metavar='L')
parser.add_argument('--complib', type=str, default='blosc:lz4hc', help='Compression library', metavar='L')
args = parser.parse_args()

assert args.train <= 1, 'Train weight must be in range [0, 1]'

classes = np.array([i.encode('ascii') for i in json.load(open(os.path.join(args.directory, args.classes)))])
colors = np.array([(i, i, i, 1) for i in range(1, len(classes) + 1)], dtype='uint8')

filters = tables.Filters(complevel=args.complevel, complib=args.complib)
if os.path.exists(args.file):
	file = tables.open_file(args.file, mode='r+')
else:
	file = tables.open_file(args.file, mode='w')
	for i in ('image', 'mask', 'class_id'):
		file.create_group(file.root, i)
	file.create_carray(file.root, 'count', atom=tables.IntAtom(), shape=(1,), filters=filters)
	file.create_earray(file.root, 'classes', atom=tables.StringAtom(25), shape=(0,), filters=filters)
	file.create_earray(file.root, 'train', atom=tables.IntAtom(), shape=(0,), filters=filters)
	file.create_earray(file.root, 'test', atom=tables.IntAtom(), shape=(0,), filters=filters)
	file.root.count[0] = 0

unique(file.root.classes, classes)

files = list(filter(lambda x: not x.endswith('.mask.png'), glob.iglob(os.path.join(args.directory, '**/*.png'), recursive=True)))
random.shuffle(files)

result, tn, queue, lock, is_done, threads = [], ceil(len(files) / args.threads), Queue(), threading.Lock(), Wrapper(False), []

writer_thread = threading.Thread(target=writer, args=(file, filters, queue, is_done, lock))
writer_thread.start()

for i in range(args.threads):
	threads.append(threading.Thread(target=worker, args=(i, files[i * tn:(i + 1) * tn], result, classes, file.root.classes[:], colors, i * tn, lock, queue)))
	threads[-1].start()

for i in threads:
	i.join()

for i in threads:
	print()

lock.acquire()
is_done.value = True
lock.release()

print('Writing files...')
writer_thread.join()

result = np.array(result) + file.root.count[0]
file.root.count[0] += len(files)

k = int(len(files) * args.train)
file.root.train.append(result[:k:])
file.root.test.append(result[k::])
