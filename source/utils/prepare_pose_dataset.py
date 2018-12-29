#!/usr/bin/python
import os
import glob
import tables
import argparse
import skimage.io
import numpy as np
from tqdm import tqdm
from utils import cut_string

parser = argparse.ArgumentParser(description='Prepare pose dataset')
parser.add_argument('directory', metavar='DIR', help='Path to dataset', type=str)
parser.add_argument('-f', '--file', type=str, default='dataset.hdf5', help='HDF5 dataset file', metavar='F')
parser.add_argument('--train', metavar='P', help='Train weight', type=float, default=0.8)
parser.add_argument('--complevel', type=int, default=9, help='Compression level', metavar='L')
parser.add_argument('--complib', type=str, default='blosc:lz4hc', help='Compression library', metavar='L')
args = parser.parse_args()

assert 0 <= args.train <= 1, 'Train weight must be in range [0, 1]'

poses = glob.glob(os.path.join(args.directory, '**/poses.txt'), recursive=True)
posestxt = [np.loadtxt(i) for i in poses]
images = [glob.iglob(os.path.join(os.path.dirname(i), '*.png')) for i in poses]

filters = tables.Filters(complevel=args.complevel, complib=args.complib)
if os.path.exists(args.file):
	file = tables.open_file(args.file, mode='r+')
else:
	file = tables.open_file(args.file, mode='w')
	for i in ('image', 'position', 'orientation'):
		file.create_group(file.root, i)
	file.create_carray(file.root, 'count', atom=tables.IntAtom(), shape=(1,), filters=filters)
	file.create_earray(file.root, 'train', atom=tables.IntAtom(), shape=(0,), filters=filters)
	file.create_earray(file.root, 'test', atom=tables.IntAtom(), shape=(0,), filters=filters)
	file.root.count[0] = 0

count = file.root.count[0]
skipped = 0

with tqdm(total=sum(i.shape[0] for i in posestxt)) as pbar:
	for pose_path, pose, files in zip(poses, posestxt, images):
		pbar.set_description(cut_string(os.path.dirname(pose_path)))
		for path in files:
			img = skimage.io.imread(path)
			if img.any():
				p = pose[int(os.path.splitext(os.path.basename(path))[0])]
				id = '_' + str(count)
				file.create_carray(file.root.image, id, obj=img, filters=filters)
				file.create_carray(file.root.position, id, obj=p[:3], filters=filters)
				file.create_carray(file.root.orientation, id, obj=p[3:], filters=filters)
				count += 1
			else:
				skipped += 1
			pbar.update(1)

ncount = count - file.root.count[0]
result = np.array(range(ncount)) + file.root.count[0]
file.root.count[0] = count

np.random.shuffle(result)
k = int(ncount * args.train)
file.root.train.append(result[:k:])
file.root.test.append(result[k::])
