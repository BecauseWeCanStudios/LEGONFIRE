#!/usr/bin/python
import os
import json
import glob
import utils
import random
import tables
import argparse
import skimage.io
import numpy as np
import multiprocessing
from tqdm import tqdm
from math import ceil
from skimage.morphology import remove_small_objects


def start_process(target, args):
	process = multiprocessing.Process(target=target, args=args, daemon=True)
	process.start()
	return process


def unique(uclasses, classes):
	new, uc = [], uclasses[:]
	for i in classes:
		if i not in uc and i not in new:
			new.append(i)
	if new:
		new.sort()
		uclasses.append(new)


def worker(n, files, classes, uclasses, colors, lock, conn):
	pbar = tqdm(files, position=n)
	uclasses = list(uclasses)
	n_colors, n_classes = len(colors), len(classes)
	for file in pbar:
		pbar.set_description(utils.cut_string(file))
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
				conn.send((
					file,
					('image', skimage.io.imread(file)[:, :, :3]),
					('mask', mask),
					('class_id', np.array([uclasses.index(i) + 1 for i in classes[ind]], dtype='uint8'))
				))
	pbar.close()


def writer(file, filters, conn, lock):
	count, polled, paths = file.root.count[0], True, []
	while multiprocessing.active_children() or polled:
		polled = conn.poll(1)
		if polled:
			path, *rest = conn.recv()
			for i in rest:
				category, value = i
				arr = file.create_carray(file.root[category], '_' + str(count), obj=value, filters=filters)
			paths.append(path)
			count += 1
	return paths


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Prepare dataset')
	parser.add_argument('directory', metavar='DIR', help='Path to dataset', type=str)
	parser.add_argument('-f', '--file', type=str, default='dataset.hdf5', help='HDF5 dataset file', metavar='F')
	parser.add_argument('--train', metavar='P', help='Train weight', type=float, default=0.8)
	parser.add_argument('--classes', metavar='C', help='Path to classes json', default='types.json')
	parser.add_argument('-p', '--processes', type=int, default=1, help='Number of processes', metavar='N')
	parser.add_argument('--complevel', type=int, default=9, help='Compression level', metavar='L')
	parser.add_argument('--complib', type=str, default='blosc:lz4hc', help='Compression library', metavar='L')
	parser.add_argument('--save_path', action='store_true', help='Save image path')
	args = parser.parse_args()

	assert 0 <= args.train <= 1, 'Train weight must be in range [0, 1]'

	classes = np.array([i.encode('ascii') for i in json.load(open(os.path.join(args.directory, args.classes)))])
	colors = np.array([i for i in range(1, len(classes) + 1)], dtype='uint8')

	filters = tables.Filters(complevel=args.complevel, complib=args.complib)
	file = utils.open_or_create_dataset_file(args.file, filters, ('image', 'mask', 'class_id'), True)

	unique(file.root.classes, classes)

	files = list(filter(lambda x: not x.endswith('.mask.png'), glob.iglob(os.path.join(args.directory, '**/*.png'), recursive=True)))
	count = len(files)
	random.shuffle(files)

	pn, lock, processes = ceil(count / args.processes), multiprocessing.Lock(), []
	conn_in, conn_out = multiprocessing.Pipe(False)

	for i in range(args.processes):
		processes.append(start_process(worker, (i, files[i * pn:(i + 1) * pn], classes, file.root.classes[:], colors, lock, conn_out)))

	files = writer(file, filters, conn_in, lock)

	for i in processes:
		print()

	if args.save_path:
		utils.save_path(file, filters, files)
	utils.split_dataset(file, len(files), args.train)
