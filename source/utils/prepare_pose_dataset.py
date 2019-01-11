#!/usr/bin/python
import os
import glob
import utils
import tables
import argparse
import skimage.io
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Prepare pose dataset')
parser.add_argument('directory', metavar='DIR', help='Path to dataset', type=str)
parser.add_argument('-f', '--file', type=str, default='dataset.hdf5', help='HDF5 dataset file', metavar='F')
parser.add_argument('--train', metavar='P', help='Train weight', type=float, default=0.8)
parser.add_argument('--complevel', type=int, default=9, help='Compression level', metavar='L')
parser.add_argument('--complib', type=str, default='blosc:lz4hc', help='Compression library', metavar='L')
parser.add_argument('--classes', type=str, help='List of classes', nargs='+', default=['1x1', '1x2', '1x3'], metavar='C')
args = parser.parse_args()

assert 0 <= args.train <= 1, 'Train weight must be in range [0, 1]'

poses = glob.glob(os.path.join(args.directory, '**/poses.txt'), recursive=True)
posestxt = [np.loadtxt(i) for i in poses]
images = [glob.iglob(os.path.join(os.path.dirname(i), '*.png')) for i in poses]

filters = tables.Filters(complevel=args.complevel, complib=args.complib)
file = utils.open_or_create_dataset_file(args.file, filters, ('image', 'val'), False)

count = file.root.count[0]
skipped = 0

with tqdm(total=sum(i.shape[0] for i in posestxt)) as pbar:
	for pose_path, pose, files in zip(poses, posestxt, images):
		dirname = os.path.dirname(pose_path)
		pbar.set_description(utils.cut_string(dirname))
		class_id = args.classes.index(os.path.split(dirname)[1])
		for path in files:
			img = skimage.io.imread(path)
			if img.any():
				img = img.reshape((*img.shape, 1))
				id = '_' + str(count)
				file.create_carray(file.root.image, id, obj=img, filters=filters)
				file.create_carray(file.root.val, id, obj=np.concatenate([pose[int(os.path.splitext(os.path.basename(path))[0])], [class_id]]))
				count += 1
			else:
				skipped += 1
			pbar.update(1)

utils.split_dataset(file, count - file.root.count[0], args.train)
