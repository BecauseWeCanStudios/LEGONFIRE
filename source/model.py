import os
import skimage.io
from mrcnn import utils
from mrcnn import config
from dataset import Dataset
import mrcnn.model as modellib

class Config(config.Config):

  NAME = 'LEGOVNO'
  IMAGES_PER_GPU = 2
  GPU_COUNT = 1
  NUM_CLASSES = 4
  STEPS_PER_EPOCH = 100
  DETECTION_MIN_CONFIDENCE = 0.9
  BACKBONE = 'resnet101'	
  IMAGE_MIN_DIM = 448
  IMAGE_MAX_DIM = 448

class InferenceConfig(Config):

  IMAGES_PER_GPU = 1


class Model:

	TRAIN = 0
	INFERENCE = 1
	COCO_WEIGHTS_PATH = './mask_rcnn_coco.h5'

	WEIGHT_LOADERS = {
		'coco': lambda self: self.__load_coco(),
		'last': lambda self: self.model.find_last()[1],
		'imagenet': lambda self: self.model.get_imagenet_weights()
	}

	def __init__(self, weights, mode, logs='./logs'):
		assert mode in (self.TRAIN, self.INFERENCE), 'Unrecognised mode'

		self.config = Config() if mode == self.TRAIN else InferenceConfig()
		self.model = modellib.MaskRCNN(mode='training' if mode == self.TRAIN else 'inference', 
			config=self.config, model_dir=logs)

		lweights = weights.lower()
		weights_path = self.WEIGHT_LOADERS[lweights](self) if lweights in self.WEIGHT_LOADERS else weights

		self.model.load_weights(weights_path, by_name=True, 
			exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'] if lweights == 'coco' else [])

	def train(self, data, epochs=30, learning_rate=1e-3):
		train_dataset = Dataset.load_and_prepare(data.root.train[:], data)
		test_dataset = Dataset.load_and_prepare(data.root.test[:], data)
		self.model.train(train_dataset, test_dataset, learning_rate=learning_rate, epochs=epochs, layers='heads')

	def detect(self, image, verbose=1):
		return self.model.detect([image], verbose=verbose)[0]

	def detect_file(self, path, verbose=1):
		return self.detect(skimage.io.imread(path), verbose)

	def __load_coco(self):
		if not os.path.exists(self.COCO_WEIGHTS_PATH):
			utils.download_trained_weights(self.COCO_WEIGHTS_PATH)
		return self.COCO_WEIGHTS_PATH

