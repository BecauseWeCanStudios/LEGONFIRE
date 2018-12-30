import keras
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

class PoseEstimationDataset(keras.utils.Sequence):

	def __init__(self, ids, data, batch_size):
		self.ids, self.data, self.batch_size = list(map(lambda x: '_' + str(x), ids)), data.root, batch_size
		self.len = int(np.ceil(len(self.ids) / batch_size))

	def __len__(self):
		return self.len

	@staticmethod
	def __get_data__(data, ids):
		return np.array([data[i][:] for i in ids])

	def __getitem__(self, index):
		ids = self.ids[index * self.batch_size : (index + 1) * self.batch_size]
		return \
				self.__get_data__(self.data.image, ids), \
				[
					self.__get_data__(self.data.position, ids), 
					self.__get_data__(self.data.orientation, ids)
				]
