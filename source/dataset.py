import numpy as np
from mrcnn import utils

class Dataset(utils.Dataset):

	NAME = 'LEGOVNO'

	@staticmethod
	def load_and_prepare(ids, data):
		result = Dataset()
		result.load_dataset_info(ids, data)
		result.prepare()
		return result

	def load_dataset_info(self, ids, data):
		self.data = data
		classes = data.root.classes[:]
		for i in range(classes.shape[0]):
			self.add_class(self.NAME, i + 1, classes[i].decode('ascii'))
		for i in ids:
			self.add_image(self.NAME, path=str(i), image_id=i, id='_' + str(i))

	def load_mask(self, image_id):
		image_info = self.image_info[image_id]
		if image_info['source'] != self.NAME:
			return super(self.__class__, self).load_mask(image_id)
		image_id = image_info['id']
		return self.data.root.mask[image_id][:], self.data.root.class_id[image_id][:]

	def load_image(self, image_id):
		image_info = self.image_info[image_id]
		if image_info['source'] != self.NAME:
			return super(self.__class__, self).load_mask(image_id)
		return self.data.root.image[image_info['id']][:]

	def image_reference(self, image_id):
		info = self.image_info[image_id]
		if info['source'] != self.NAME:
			super(self.__class__, self).image_reference(image_id)
		else:
			return self.NAME
