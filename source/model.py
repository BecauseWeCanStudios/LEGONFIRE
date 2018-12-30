import os
import keras
import skimage.io
import keras_contrib.applications
from mrcnn import utils
from mrcnn import config
from dataset import Dataset, PoseEstimationDataset
import numpy as np
import keras.backend as K
import mrcnn.model as modellib

class Config(config.Config):

  NAME = 'LEGOVNO'
  IMAGES_PER_GPU = 1
  GPU_COUNT = 1
  NUM_CLASSES = 4
  STEPS_PER_EPOCH = 1000
  DETECTION_MIN_CONFIDENCE = 0.9
  BACKBONE = 'resnet101'	
  IMAGE_MIN_DIM = 1024
  IMAGE_MAX_DIM = 1024

class InferenceConfig(Config): 
	pass

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
		self.model.train(train_dataset, test_dataset, learning_rate=learning_rate, epochs=epochs, layers='all')

	def detect(self, image, verbose=1):
		return self.model.detect([image], verbose=verbose)[0]

	def detect_file(self, path, verbose=1):
		return self.detect(skimage.io.imread(path), verbose)

	def __load_coco(self):
		if not os.path.exists(self.COCO_WEIGHTS_PATH):
			utils.download_trained_weights(self.COCO_WEIGHTS_PATH)
		return self.COCO_WEIGHTS_PATH

class ActivationLayer(keras.engine.topology.Layer):

	def __init__(self, **kwargs):
		super(ActivationLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		super(ActivationLayer, self).build(input_shape)

	def call(self, x):
		return x / K.sqrt(K.sum(K.pow(x, 2)))

	def compute_output_shape(self, input_shape):
		return (input_shape[0], 4)

def PoseEstimationLoss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))

class PoseEstimationConfig:

	BACKBONE = 'resnet18'
	INPUT_SHAPE = (750, 1000, 1)
	SHARED_LAYERS = 1
	SHARED_UNITS = 256
	POSITION_LAYERS = 1
	POSITION_UNITS = 256
	ORIENTATION_LAYERS = 1
	ORIENTATION_UNITS = 256
	BATCH_SIZE = 1
	VALIDATION_BATCH_SIZE = 1
	OPTIMIZER = keras.optimizers.Adam(1e-3)
	LOSSES = [PoseEstimationLoss, PoseEstimationLoss]
	LOSS_WEIGHTS = [1, 1]
	SAVE_WEIGHTS_PATH = './logs'
	SAVE_PERIOD = 1
	STEPS_PER_EPOCH = None
	VALIDATION_STEPS = None

class PoseEstimationModel():

	BACKBONES = {
		'resnet18': lambda input_shape:
			PoseEstimationModel.__resnet(input_shape, 'basic', [2, 2, 2, 2]),
		'resnet34': lambda input_shape:
			PoseEstimationModel.__resnet(input_shape, 'basic', [3, 4, 6, 3]),
		'resnet50': lambda input_shape:
			PoseEstimationModel.__resnet(input_shape, 'bottleneck', [3, 4, 6, 3]),
		'xception': lambda input_shape:
			keras.applications.xception.Xception(include_top=False, weights=None, input_shape=input_shape, classes=None)
	}

	def __init__(self, config):
		shared_model = keras.models.Sequential()
		shared_model.add(PoseEstimationModel.BACKBONES[config.BACKBONE](config.INPUT_SHAPE))
		shared_model.add(keras.layers.Flatten())

		for i in range(config.SHARED_LAYERS):
			shared_model.add(keras.layers.Dense(config.SHARED_UNITS, activation='relu'))

		model = keras.models.Model(inputs=[shared_model.input], outputs=[
				PoseEstimationModel.__make_fc_layers(shared_model.output, config.POSITION_LAYERS, config.POSITION_UNITS, 3), 
				ActivationLayer()(PoseEstimationModel.__make_fc_layers(shared_model.output, config.ORIENTATION_LAYERS, config.ORIENTATION_UNITS, 4))
		])

		model.compile(optimizer=config.OPTIMIZER, loss=config.LOSSES, loss_weights=config.LOSS_WEIGHTS,)

		self.model, self.config = model, config

	def train(self, data, epochs):
		train_dataset = PoseEstimationDataset(data.root.train[:], data, self.config.BATCH_SIZE)
		test_dataset = PoseEstimationDataset(data.root.test[:], data, 
			self.config.BATCH_SIZE if self.config.BATCH_SIZE else self.config.VALIDATION_BATCH_SIZE)
		save_best = keras.callbacks.ModelCheckpoint(
			os.path.join(self.config.SAVE_WEIGHTS_PATH, 'weights.{epoch:04d}.hdf5'), 
			verbose=0, 
			save_weights_only=True, 
			period=self.config.SAVE_PERIOD
		)
		self.model.fit_generator(
			train_dataset, 
			validation_data=test_dataset,
			steps_per_epoch=self.config.STEPS_PER_EPOCH, 
			epochs=epochs, 
			callbacks=[save_best], 
			shuffle=False, 
			workers=0,
			validation_steps=self.config.VALIDATION_STEPS
		)

	@staticmethod
	def __make_fc_layers(inputs, count, units, last_units):
		assert count > 0
		for i in range(count - 1):
			inputs = keras.layers.Dense(units, activation='relu')(inputs)
		return keras.layers.Dense(last_units)(inputs)

	@staticmethod
	def __resnet(input_shape, block, repetitions):
		return keras_contrib.applications.resnet.ResNet(input_shape, None, block, repetitions=repetitions, include_top=False)
		

