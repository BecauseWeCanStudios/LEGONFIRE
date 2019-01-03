#!/usr/bin/python
import os
import json
import glob
import utils
import tables
import argparse
import skimage.io
import skimage.draw
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Add annotated photos')
parser.add_argument('directory', metavar='DIR', help='Path to photos', type=str)
parser.add_argument('-f', '--file', type=str, default='dataset.hdf5', help='HDF5 dataset file', metavar='F')
parser.add_argument('--train', metavar='P', help='Train weight', type=float, default=0.8)
parser.add_argument('--complevel', type=int, default=9, help='Compression level', metavar='L')
parser.add_argument('--complib', type=str, default='blosc:lz4hc', help='Compression library', metavar='L')
parser.add_argument('--save_path', action='store_true', help='Save image path')
args = parser.parse_args()

assert 0 <= args.train <= 1, 'Train weight must be in range [0, 1]'

filters = tables.Filters(complevel=args.complevel, complib=args.complib)
file = utils.open_or_create_dataset_file(args.file, filters, ('image', 'mask', 'class_id'), True)

classes = file.root.classes[:]
new_classes = []
count = file.root.count[0]

annotations = glob.glob(os.path.join(args.directory, '**/*.json'), recursive=True)
files = []

apbar = tqdm(annotations, position=0)
for annotation in apbar:
	apbar.set_description(utils.cut_string(annotation))

	path = os.path.dirname(annotation)
	annotation = json.load(open(annotation))

	pbar = tqdm(list(annotation['_via_img_metadata'].values()), position=1)
	for data in pbar:
		pbar.set_description(utils.cut_string(data['filename']))

		img_path = os.path.join(path, data['filename'])
		img = skimage.io.imread(img_path)

		l = len(data['regions'])
		mask = np.zeros(shape=(*img.shape[:-1], l)).astype(bool)
		class_ids = np.ndarray(l, dtype='uint8')

		for i in range(l):
			rd = data['regions'][i]
			c, r = rd['shape_attributes']['all_points_x'], rd['shape_attributes']['all_points_y']
			rr, cc = skimage.draw.polygon(r, c)
			mask[rr, cc, i] = True

			id = rd['region_attributes']['size'].encode('ascii')
			if id not in classes:
				if id in new_classes:
					class_ids[i] = len(classes) + new_classes.index(id)
				else:
					new_classes.append(id)
			else:
				class_ids[i] = classes.index(id) + 1

		id = '_' + str(count)
		count += 1
		file.create_carray(file.root.image,    id, obj=img,       filters=filters)
		file.create_carray(file.root.mask,     id, obj=mask,      filters=filters)
		file.create_carray(file.root.class_id, id, obj=class_ids, filters=filters)
		files.append(img_path)

print('\n')

new_classes.sort()
file.root.classes.append(new_classes)

if args.save_path:
	utils.save_path(file, filters, files)
utils.split_dataset(file, len(files), args.train)
