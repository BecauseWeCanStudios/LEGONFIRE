#!/usr/bin/python
import os
import json
import glob
import random
import tables
import argparse
import threading
import skimage.io
import numpy as np
from tqdm import tqdm
from math import ceil
from utils import cut_string
from queue import Queue, Empty
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects


def unique(uclasses, classes):
	new, uc = [], uclasses[:]
	for i in classes:
		if i not in uc and i not in new:
			new.append(i)
	if new:
		new.sort()
		uclasses.append(new)


def worker(n, files, classes, uclasses, colors, count, lock, queue):
	pbar = tqdm(files, position=n)
	uclasses = ['background'] + list(set(classes))
	n_colors, n_classes = len(colors), len(classes)
	for file in pbar:
		pbar.set_description(cut_string(file))
		dir, name = os.path.dirname(file), os.path.splitext(os.path.basename(file))[0]
		mask = skimage.io.imread(os.path.join(dir, name + '.mask.png'))
		mask = mask.reshape((*mask.shape, 1)) if len(mask.shape) <= 2 else mask[:, :, [0]]
		mask = mask.repeat(n_colors, axis=2) == colors
		ind = [i for i in range(n_classes) if mask[::, ::, i].any()]
		if ind:
			mask = mask[:, :, ind]
			for i in range(mask.shape[-1]):
				remove_small_objects(mask[:, :, i], connectivity=2, in_place=True)
			with lock:
				queue.put_nowait
				queue.put_nowait(('image', count, skimage.io.imread(file)[:, :, :3]))
				queue.put_nowait(('mask', count, mask))
				queue.put_nowait(('class_id', count, np.array([uclasses.index(i) + 1 for i in classes[ind]], dtype='uint8')))
			count += 1


def writer(file, filters, queue, event, lock):
	count = file.root.count[0]
	while True:
		b = event.wait(1)
		with lock:
			a = []
			while not queue.empty():
				a.append(queue.get_nowait())
		for i in a:
			category, id, value = i
			id += count
			arr = file.create_carray(file.root[category], '_' + str(id), obj=value, filters=filters)
		if b:
			return


parser = argparse.ArgumentParser(description='Prepare dataset')
parser.add_argument('directory', metavar='DIR', help='Path to dataset', type=str)
parser.add_argument('-f', '--file', type=str, default='dataset.hdf5', help='HDF5 dataset file', metavar='F')
parser.add_argument('--train', metavar='P', help='Train weight', type=float, default=0.8)
parser.add_argument('--classes', metavar='C', help='Path to classes json', default='types.json')
parser.add_argument('-t', '--threads', type=int, default=1, help='Number of threads', metavar='N')
parser.add_argument('--complevel', type=int, default=9, help='Compression level', metavar='L')
parser.add_argument('--complib', type=str, default='blosc:lz4hc', help='Compression library', metavar='L')
parser.add_argument('--save_path', action='store_true', help='Save image path')
args = parser.parse_args()

assert args.train <= 1, 'Train weight must be in range [0, 1]'

classes = np.array([i.encode('ascii') for i in json.load(open(os.path.join(args.directory, args.classes)))])
colors = np.array([i for i in range(1, len(classes) + 1)], dtype='uint8')

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
count = len(files)
random.shuffle(files)

tn, queue, lock, event, threads = ceil(count / args.threads), Queue(), threading.Lock(), threading.Event(), []

writer_thread = threading.Thread(target=writer, args=(file, filters, queue, event, lock))
writer_thread.start()

for i in range(args.threads):
	threads.append(threading.Thread(target=worker, args=(i, files[i * tn:(i + 1) * tn], classes, file.root.classes[:], colors, i * tn, lock, queue)))
	threads[-1].start()

for i in threads:
	i.join()

for i in threads:
	print()

print('Writing files...')
event.set()
writer_thread.join()

if args.save_path:
	if not '/path' in file.root:
		file.create_earray(file.root, 'path', atom=tables.StringAtom(50), shape=(0,), filters=filters)
	diff = file.root.count[0] - file.root.path.shape[0]
	if diff:
		file.root.path.append(np.full((diff,), ''))
	file.root.path.append(np.fromiter((i[-50:] for i in files), dtype='|S50'))

result = np.array(range(count)) + file.root.count[0]
file.root.count[0] += count

k = int(count * args.train)
file.root.train.append(result[:k:])
file.root.test.append(result[k::])
