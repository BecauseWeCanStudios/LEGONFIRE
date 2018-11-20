import skimage
from mrcnn import utils
from mrcnn import config
from dataset import Dataset
import mrcnn.model as modellib

class Config(config.Config):

  NAME = 'LEGOVNO'
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


class Model:

	TRAIN = 0
	INFERENCE = 1
	COCO_WEIGHTS_PATH = './mask_rcnn_coco.h5'

	WEIGHT_LOADERS = {
		'coco': __load_coco,
		'last': lambda self: self.model.find_last()[1],
		'imagenet': lambda self: self.model.get_imagenet_weights()
	}

	def __init__(self, weights, mode, logs='./logs'):
		assert mode in (TRAIN, INFERENCE), 'Unrecognised mode'

		self.config = Config() if mode == TRAIN else InferenceConfig
		self.model = modellib.MaskRCNN(mode='training' if mode == TRAIN else 'inference', config=self.config, model_dir=logs)

		lweights = weights.lower()
		weights_path = WEIGHT_LOADERS[lweights](self) if lweights in WEIGHT_LOADERS else weights

		model.load_weights(weights_path, by_name=True, 
			exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'] if lweights == 'coco' else [])

	def train(self, train_dir, test_dir=None, epochs=30, learning_rate=1e-3):
		train_dataset = Dataset.load_and_prepare(train_dir)
		test_dataset = Dataset.load_and_prepare(test_dir) if test_dir else train_dataet
		self.model.train(train_dataset, test_dataset, learning_rate=learning_rate, epochs=epochs, layers='heads')

	def detect(self, image, verbose=1):
		return self.model.detect([image], verbose=verbose)[0]

	def detect_file(self, path, verbose=1):
		return self.detect(skimage.io.imread(path), verbose)

	def __load_coco(self):
		utils.download_trained_weights(COCO_WEIGHTS_PATH)
		return COCO_WEIGHTS_PATH

