#!/usr/bin/python
import tqdm
import tables
import argparse
import numpy as np
import matplotlib.pyplot as plt
from camera import random_colors
from model import PoseEstimationModel
from dataset import PoseEstimationDataset

parser = argparse.ArgumentParser(description='Plot loss and metrics histograms')
parser.add_argument('-d', '--dataset', help='Path to HDF5 file containing dataset', type=str, default='dataset.hdf5')
parser.add_argument('-p', '--part', help='Train or test part of dataset', type=str, choices=['train', 'test'], default='train')
parser.add_argument('-w', '--weights', help='Path to weights', type=str)
parser.add_argument('-c', '--cols', help='Number of columns', type=int, default=2)
parser.add_argument('-b', '--bins', help='Number of bins', type=int, default=50)
args = parser.parse_args()

f = tables.open_file(args.dataset)
dataset = PoseEstimationDataset(f.root.train[:] if args.part == 'train' else f.root.test[:], f, 1)
model = PoseEstimationModel(weights=args.weights)
res = []

for x, y in tqdm.tqdm(dataset):
	res.append(model.evaluate(x, y, batch_size=1))

res = np.array(res)
names = model.model.metrics_names
rows = len(names) // args.cols
colors = np.array(random_colors(len(names))) / 256

for i in range(len(names)):
	plt.subplot(rows, args.cols, i + 1)
	plt.hist(res[..., i], bins=args.bins, color=colors[i])
	plt.title(names[i])

plt.show()
