import numpy as np
from mrcnn import utils

class Dataset(utils.Dataset):

	NAME = 'LEGOVNO'

	@staticmethod
	def load_and_prepare(ids, data, classes):
		result = Dataset()
		result.load_dataset_info(ids, data, classes)
		result.prepare()
		return result

	def load_dataset_info(self, ids, data, classes):
		self.data = data
		for i in range(len(classes)):
			self.add_class(self.NAME, i + 1, classes[i])
		for i in ids:
			self.add_image(
				self.NAME,
				path=i,
				image_id=i, 
				image='image{}'.format(i), 
				mask='mask{}'.format(i),
				class_ids=np.array([classes.index(i.decode('ascii')) + 1 for i in data.root['classes{}'.format(i)][:]], dtype=np.int32)
			)

	def load_mask(self, image_id):
		image_info = self.image_info[image_id]
		if image_info['source'] != self.NAME:
			return super(self.__class__, self).load_mask(image_id)
		return self.data.root[image_info['mask']][:], image_info['class_ids']

	def load_image(self, image_id):
		image_info = self.image_info[image_id]
		if image_info['source'] != self.NAME:
			return super(self.__class__, self).load_mask(image_id)
		return self.data.root[image_info['image']][:]

	def image_reference(self, image_id):
		info = self.image_info[image_id]
		if info['source'] != self.NAME:
			super(self.__class__, self).image_reference(image_id)
		else:
			return self.NAME
