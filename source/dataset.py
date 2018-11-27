import os
import glob
import json
import psutil
import random
import skimage
import numpy as np
from mrcnn import utils

class Dataset(utils.Dataset):

	NAME = 'LEGOVNO'

	@staticmethod
	def load_and_prepare(dir, classes):
		result = Dataset()
		result.load_dataset_info(dir, classes)
		result.prepare()
		return result

	def load_dataset_info(self, file, classes):
		for i in range(len(classes)):
			self.add_class(self.NAME, i + 1, classes[i])
		dataset, id = json.load(open(file)), 0
		for i in dataset:
			image = skimage.io.imread(i['image'])
			bmask = np.load(i['boolmask'])
			self.add_image(self.NAME, image_id=id, path=i['image'], 
				image=image, mask=bmask['mask'], 
				class_ids=np.array([classes.index(i) + 1 for i in bmask['classes']]), dtype=np.int32)
			id += 1

	def load_mask(self, image_id):
		image_info = self.image_info[image_id]
		if image_info['source'] != self.NAME:
			return super(self.__class__, self).load_mask(image_id)
		return image_info['mask'], image_info['class_ids']

	def load_image(self, image_id):
		image_info = self.image_info[image_id]
		if image_info['source'] != self.NAME:
			return super(self.__class__, self).load_mask(image_id)
		return image_info['image']

	def image_reference(self, image_id):
		info = self.image_info[image_id]
		if info['source'] != self.NAME:
			super(self.__class__, self).image_reference(image_id)
		else:
			return info['path']
