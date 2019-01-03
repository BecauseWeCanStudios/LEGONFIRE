import os
import tables
import numpy as np

def make_dirs(path):
	path = os.path.dirname(path)
	try:
		if not os.path.exists(path):
			os.makedirs(path)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

def cut_string(s, n=20):
	if len(s) > n:
		return '...' + s[3 - n::]
	return ' ' * (n - len(s)) + s

def open_or_create_dataset_file(filename, filters, groups, add_classes):
	if os.path.exists(filename):
		return tables.open_file(filename, mode='r+')
	file = tables.open_file(filename, mode='w')
	for i in groups:
		file.create_group(file.root, i)
	file.create_carray(file.root, 'count', atom=tables.IntAtom(), shape=(1,), filters=filters)
	if add_classes:
		file.create_earray(file.root, 'classes', atom=tables.StringAtom(25), shape=(0,), filters=filters)
	file.create_earray(file.root, 'train', atom=tables.IntAtom(), shape=(0,), filters=filters)
	file.create_earray(file.root, 'test', atom=tables.IntAtom(), shape=(0,), filters=filters)
	file.root.count[0] = 0
	return file

def save_path(file, filters, files):
	if not '/path' in file.root:
		file.create_earray(file.root, 'path', atom=tables.StringAtom(50), shape=(0,), filters=filters)
	diff = file.root.count[0] - file.root.path.shape[0]
	if diff:
		file.root.path.append(np.full((diff,), ''))
	file.root.path.append(np.fromiter((i[-50:] for i in files), dtype='|S50'))

def split_dataset(file, count, ratio):
	result = np.array(range(count)) + file.root.count[0]
	np.random.shuffle(result)

	file.root.count[0] += count

	k = int(count * ratio)
	file.root.train.append(result[:k:])
	file.root.test.append(result[k::])
	