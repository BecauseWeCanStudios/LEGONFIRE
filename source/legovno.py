#!/usr/bin/python
import os
import cv2
import json
import tables
import camera
import skimage
import argparse
import numpy as np
import mrcnn.model as modellib
from model import Model
from mrcnn import utils
from mrcnn import config
from mrcnn import visualize

def color_splash(image, mask):
	gray, mask = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255, (np.sum(mask, -1, keepdims=True) >= 1)
	if mask.shape[0] > 0:
		splash = np.where(mask, image, gray).astype(np.uint8)
	else:
		splash = gray
	return splash

def detect_and_splash(model, image):
	result = model.detect(image)
	return color_splash(image, result['masks'])

def splash_image(model, path, output_path):
	skimage.io.imsave(os.path.join(output_path, '{}_splash.png'.format(os.path.splitext(os.path.basename(path))[0])), 
		detect_and_splash(model, skimage.io.imread(path)))

def splash_video(model, path, output_path):
	vcapture = cv2.VideoCapture(path)
	vwriter = cv2.VideoWriter(
		os.path.join(output_path, '{}_splash.mp4'.format(os.path.splitext(os.path.basename(path))[0])),
		cv2.VideoWriter_fourcc(*'mp4v'),
		vcapture.get(cv2.CAP_PROP_FPS), 
		(int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
	)
	count, success = 0, True
	while success:
		success, image = vcapture.read()
		if success:
			vwriter.write(detect_and_splash(model, image[..., ::-1])[..., ::-1])
			count += 1
			vwriter.release()

def visualize_image(model, path):
	image = skimage.io.imread(path)
	result = model.detect(image)
	visualize.display_instances(image, result['rois'], result['masks'], result['class_ids'], 
		['BG', '1x1', '1x2', '1x3'], result['scores'])

OPERATIONS = {
	'train': lambda model, args: model.train(
					tables.open_file(args.dataset, mode='r'),
					json.load(open(args.train)), 
					json.load(open(args.test)),
					['1x1', '1x2', '1x3'], 
					args.epochs, 
					args.learning_rate
				),
	'splash_image': lambda model, args: splash_image(model, args.input, args.output),
	'splash_video': lambda model, args: splash_video(model, args.input, args.output),
	'visualize': lambda model, args: visualize_image(model, args.input),
	'splash_camera': lambda model, args: camera.start(model, ['BG', '1x1', '1x2', '1x3'], (448, 448))
}

parser = argparse.ArgumentParser(description='Program ti die yourself')
parser.add_argument('operation', metavar='OP', help='Operation to be executed', 
	choices=('train', 'splash_image', 'splash_video', 'splash_camera', 'visualize'))
parser.add_argument('-d', '--dataset', help='Path to HDF5 file containing dataset', type=str, default='dataset.hdf5')
parser.add_argument('-t', '--train', help='Path to JSON file containing train ids', type=str, default='train.json')
parser.add_argument('-v', '--test', help='Path to JSON file containing test ids', type=str, default='test.json')
parser.add_argument('-w', '--weights', help='Path to .h5 weights file', type=str, default='coco')
parser.add_argument('-l', '--learning_rate', help='Learning rate', type=float, default=1e-3)
parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, default=30)
parser.add_argument('--logs', help='Logs file', default='./logs', type=str)
parser.add_argument('-i', '--input', help='Path or URL to image or video', type=str)
parser.add_argument('-o', '--output', help='Path to output directory', type=str)
args = parser.parse_args()

if args.operation != 'splash_camera' and args.operation != 'train':
	assert args.input, "Provide --input to apply color splash"

model = Model(args.weights, Model.TRAIN if args.operation == 'train' else Model.INFERENCE, logs=args.logs)
model.config.display()

OPERATIONS[args.operation](model, args)
