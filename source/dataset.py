import os
import glob
import psutil
import random
import skimage
from mrcnn import utils

class Dataset(utils.Dataset):

	@staticmethod
	def load_and_prepare(dir):
		result = Dataset()
		result.load_dataset_info(dir)
		result.prepare()
		return result

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
