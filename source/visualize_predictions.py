#!/usr/bin/python
import tables
import argparse
import numpy as np
from model import PoseEstimationModel
from dataset import PoseEstimationDataset
from utils.lego_model_visualizer import LegoModelVisualizer

class Predictor:

	def __init__(self, model, dataset):
		self.model, self.dataset = model, dataset

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		x, y = self.dataset[index]
		true, pred = y[0], self.model.predict(x)[0]
		return true[-1], (true[:3], pred[:3]), (true[3:-1], pred[3:])

SELECT_DATA = {
	'train': lambda f: f.root.train[:],
	'test': lambda f: f.root.test[:],
	'all': lambda f: range(f.root.count[0])
}

parser = argparse.ArgumentParser(description='Evaluate and visualize model on dataset')
parser.add_argument('-d', '--dataset', help='Path to HDF5 file containing dataset', type=str, default='dataset.hdf5')
parser.add_argument('-p', '--part', help='Train or test part of dataset', type=str, choices=['train', 'test', 'all'], default='train')
parser.add_argument('-w', '--weights', help='Path to weights', type=str)
parser.add_argument('-m', '--models', help='Models to show', type=str, nargs='+', default=['../models/1x1.obj', '../models/1x2.obj', '../models/1x3.obj'])
args = parser.parse_args()

f = tables.open_file(args.dataset)
dataset = PoseEstimationDataset(SELECT_DATA[args.part](f), f, 1)
model = PoseEstimationModel(weights=args.weights)

LegoModelVisualizer(args.models, Predictor(model, dataset), verbose=True)
