#!/usr/bin/python
import os
import cv2
import glob
import psutil
import random
import skimage
import datetime
import argparse
import numpy as np
import mrcnn.model as modellib
from mrcnn import utils
from mrcnn import config
from mrcnn import visualize

TITLE = 'LEGOVNO'
COCO_WEIGHTS_PATH = './mask_rcnn_coco.h5'

class Config(config.Config):

  NAME = TITLE
  IMAGES_PER_GPU = 5
  GPU_COUNT = 1
  NUM_CLASSES = 4
  STEPS_PER_EPOCH = 100
  DETECTION_MIN_CONFIDENCE = 0.9
  BACKBONE = 'resnet50'	
  IMAGE_MIN_DIM = 448
  IMAGE_MAX_DIM = 448

class InferenceConfig(Config):

  IMAGES_PER_GPU = 1

class Dataset(utils.Dataset):

	def load_dataset_info(self, directory):
		directory = os.path.join(directory, '*')
		count, id, globs = 1, 0, []
		for d in glob.iglob(directory):
			if not os.path.isdir(d):
				continue
			self.add_class(TITLE, count, os.path.basename(d))
			shuffled = glob.glob(os.path.join(d, '**/*.png'), recursive=True)
			random.shuffle(shuffled)
			globs.append((i for i in shuffled))
			count += 1

		index = 0
		read_masks = {}

		while psutil.virtual_memory().available > 500 * 1024 * 1024:
			try:
				path = next(globs[index])
				img = skimage.io.imread(path)
				dir_path = os.path.dirname(path)
				mask_path = os.path.splitext(os.path.basename(path))[0]
				mask_path = os.path.join(dir_path, mask_path[:mask_path.find('_')] + '.mask')
				if mask_path not in read_masks:
					mask = skimage.io.imread(mask_path).astype(bool)
					read_masks[mask_path] = mask.reshape((*mask.shape, 1))
				mask = read_masks[mask_path]
				self.add_image(TITLE, image_id=id, path=path, image=img, mask=mask, class_ids=np.array([index + 1], dtype=np.int32))
				id += 1
			except StopIteration as e:
				pass

			index = (index + 1) % len(globs)

	def load_mask(self, image_id):
		image_info = self.image_info[image_id]
		if image_info['source'] != TITLE:
			return super(self.__class__, self).load_mask(image_id)
		return image_info['mask'], image_info['class_ids']

	def load_image(self, image_id):
		image_info = self.image_info[image_id]
		if image_info['source'] != TITLE:
			return super(self.__class__, self).load_mask(image_id)
		return image_info['image']

	def image_reference(self, image_id):
		info = self.image_info[image_id]
		if info['source'] != TITLE:
			super(self.__class__, self).image_reference(image_id)
		else:
			return info['path']

def train(model, args):
	train_dataset = Dataset()
	train_dataset.load_dataset_info(args.train_dataset)
	train_dataset.prepare()

	if args.test_dataset:
		test_dataset = Dataset()
		test_dataset.load_dataset_info(args.test_dataset)
		test_dataset.prepare()
	else:
		test_dataset = train_dataset

	print('Hell begins')
	model.train(train_dataset, test_dataset, learning_rate=args.learning_rate, epochs=args.epochs, layers='heads')

def color_splash(image, mask):
	gray, mask = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255, (np.sum(mask, -1, keepdims=True) >= 1)
	if mask.shape[0] > 0:
		splash = np.where(mask, image, gray).astype(np.uint8)
	else:
		splash = gray
	return splash

def detect_and_color_splash(model, image_path=None, video_path=None, output_path='./'):
	assert image_path or video_path
	
	if image_path:
		print('Running on {}'.format(image_path))
		image = skimage.io.imread(args.image)
		r = model.detect([image], verbose=1)[0]
		visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], ['BG', '1x1', '1x2', '1x3'], r['scores'])
	elif video_path:
		vcapture = cv2.VideoCapture(video_path)
		width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = vcapture.get(cv2.CAP_PROP_FPS)
		file_name = os.path.join(output_path, './{}_splash_{:%Y%m%dT%H%M%S}.avi'.format(os.path.basename(image_path), datetime.datetime.now()))
		vwriter = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
		count, success = 0, True
		while success:
			print('Frame: ', count)
			success, image = vcapture.read()
			if success:
				image = image[..., ::-1]
				r = model.detect([image], verbose=0)[0]
				splash = color_splash(image, r['masks'])[..., ::-1]
				vwriter.write(splash)
				count += 1
				vwriter.release()

	print('Saved to ', file_name)

parser = argparse.ArgumentParser(description='Program ti die yourself')
parser.add_argument('operation', metavar='OP', help='Operation to be executed', choices=('train', 'splash'))
parser.add_argument('-t', '--train_dataset', help='Path to train dataset', type=str)
parser.add_argument('-v', '--test_dataset', help='Path to test dataset', type=str)
parser.add_argument('-w', '--weights', help='Path to .h5 weights file', type=str, default='coco')
parser.add_argument('-l', '--learning_rate', help='Learning rate', type=float, default=1e-3)
parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, default=30)
parser.add_argument('--logs', help='Logs file', default='./logs', type=str)
parser.add_argument('--image', help='Path or URL to image', type=str)
parser.add_argument('--video', help='Path or URL to video', type=str)
args = parser.parse_args()

if args.operation == 'train':
	assert args.train_dataset, 'Argument --train_dataset is required for training'
else:
	assert args.image or args.video, "Provide --image or --video to apply color splash"

conf = Config() if args.operation == 'train' else InferenceConfig()
conf.display()

model = modellib.MaskRCNN(mode='training' if args.operation == 'train' else 'inference', config=conf, model_dir=args.logs)

if args.weights.lower() == 'coco':
	weights_path = COCO_WEIGHTS_PATH
	if not os.path.exists(weights_path):
		utils.download_trained_weights(weights_path)
elif args.weights.lower() == 'last':
	weights_path = model.find_last()[1]
elif args.weights.lower() == 'imagenet':
	weights_path = model.get_imagenet_weights()
else:
	weights_path = args.weights

if args.weights.lower() == 'coco':
	model.load_weights(weights_path, by_name=True, exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])
else:
	model.load_weights(weights_path, by_name=True)

if args.operation == 'train':
	train(model, args)
else:
	detect_and_color_splash(model, image_path=args.image, video_path=args.video)
