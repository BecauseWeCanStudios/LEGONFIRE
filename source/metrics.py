import trimesh  
import numpy as np
import keras.backend as K
from math import pi

def cross(a, b):
	a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2]
	b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2]
	return K.stack([a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1], -1)

def rotate_vector_by_quaternion(q, v):
		w, q = q[..., 0], q[..., 1:]
		return 2 * K.expand_dims(K.batch_dot(v, q), 2) * K.expand_dims(q, 1) + \
				K.expand_dims(K.expand_dims(K.pow(w, 2), -1) - K.batch_dot(q, q, axes=1), 2) * v + \
				2 * K.repeat_elements(K.expand_dims(K.expand_dims(w, -1), -1), 3, axis=-1) * cross(K.expand_dims(q, 1), v)

def acos(x):
	negate = K.cast(x < 0, 'float32')
	x = K.abs(x)
	ret = K.zeros_like(x)
	for i in (-0.0187293, 0.0742610, -0.2121144):
		ret += i
		ret *= x
	ret += 1.5707288
	ret *= K.sqrt(1 - x)
	ret -= 2 * negate * ret
	return negate * pi + ret

def extract_offset(*args):
	return [i[..., :3] for i in args]

def extract_quaternion(*args):
	return [i[..., 3:] for i in args]

def Input():

	def input(true, pred, v_true, v_pred):
		return v_true, v_pred

	return input

def PassAndExtractData(base, extractor, true, pred, v_true, v_pred):
	v_true, v_pred = base(true, pred, v_true, v_pred)
	if extractor:
		true, pred = extractor(true, pred)
	return true, pred, v_true, v_pred

def BasicTranformer(extractor, transformation):

	def transformer(base):

		def transform(true, pred, v_true, v_pred):
			true, pred, v_true, v_pred = PassAndExtractData(base, extractor, true, pred, v_true, v_pred)
			return transformation(true, pred, v_true, v_pred)

		return transform

	return transformer

def RotationTransform(extractor=None):

	def rotate(true, pred, v_true, v_pred):
		return rotate_vector_by_quaternion(true, v_true), rotate_vector_by_quaternion(pred, v_pred)

	return BasicTranformer(extractor, rotate)

def OffsetTransform(extractor=None):

	def offset(true, pred, v_true, v_pred):
		return v_true + K.expand_dims(true, 1), v_pred + K.expand_dims(pred, 1)

	return BasicTranformer(extractor, offset)

def Diff(true, pred):
	return K.sum(K.square(true - pred), axis=-1)

def BaseDiff(extractor, transform):

	transform = transform(extractor)(Input())

	def diff(true, pred, vertices):
		return Diff(*transform(true, pred, vertices, vertices))

	return diff

def AngleDiff(extractor=None):
	return BaseDiff(extractor, RotationTransform)

def DistanceDiff(extractor=None):
	return BaseDiff(extractor, OffsetTransform)

def DiffSum(diff, *args):
	return K.sum(diff, axis=-1)

def DiffMean(diff, lens):
	return DiffSum(diff) / lens

def DiffMax(diff, *args):
	return K.max(diff, axis=-1)

def extract_data(y_true, meshes, mask, lens):
	y_true, classes = y_true[..., :-1], K.cast(K.round(y_true[..., -1]), 'int32')
	return y_true, classes, K.gather(meshes, classes), K.gather(mask, classes), K.gather(lens, classes)

def SequentialLoss(transformers, activation, extract_w=lambda x: x[..., 3]):

	transform = Input()
	for transformer in transformers:
		transform = transformer(transform)

	def create_loss(meshes, mask, lens):

		def loss(y_true, y_pred):
			y_true, classes, vertices, m, l = extract_data(y_true, meshes, mask, lens)
			diff = Diff(*transform(y_true, y_pred, vertices, vertices)) * m
			if extract_w:
				return K.maximum(-extract_w(y_pred), activation(diff, l))
			else:
				return activation(diff, l)

		return loss

	return create_loss

def SumLoss(diffs, weights, activation, extract_w=lambda x: x[..., 3]):

	def create_loss(meshes, mask, lens):

		def loss(y_true, y_pred):
			y_true, classes, vertices, m, l = extract_data(y_true, meshes, mask, lens)
			result = activation(diffs[0](y_true, y_pred, vertices) * m, l) * weights[0]
			for i in range(1, len(diffs)):
				result += activation(diffs[i](y_true, y_pred, vertices) * m, l) * weights[i]
			if extract_w:
				return K.maximum(-extract_w(y_pred), result)
			return result

		return loss

	return create_loss

def MeshLoss(models, loss_base):
	meshes = [trimesh.load(i) for i in models]
	for mesh in meshes:
		mesh.vertices = (mesh.vertices - mesh.center_mass) * 50
	lens = [len(i.vertices) for i in meshes]
	n = max(lens) + 1
	meshes = K.constant(np.stack([np.concatenate([np.array(i.vertices), np.zeros((n - j, 3))], axis=0) for i, j in zip(meshes, lens)]))
	mask = K.constant([[j < l for j in range(n)] for l in lens], dtype='float32')
	lens = K.constant(lens, dtype='float32')
	return loss_base(meshes, mask, lens)

def QuaternionDistanceMetric(extractor=None):

	def quaternion_distance(y_true, y_pred):
		if extractor:
			y_true, y_pred = extractor(y_true[..., :-1], y_pred)
		return K.clip(1 - K.square(K.sum(y_true * y_pred, axis=-1)), 0, 1)

	return quaternion_distance

def QuaternionAngleMetric(extractor=None):

	metric = QuaternionDistanceMetric(extractor)

	def quaternion_angle(y_true, y_pred):
		return acos(1 - 2 * metric(y_true, y_pred)) / pi * 180

	return quaternion_angle

def DistanceMetric(extractor=None):

	def distance(y_true, y_pred):
		if extractor:
			y_true, y_pred = extractor(y_true[..., :-1], y_pred)
		return K.sqrt(Diff(y_true, y_pred))

	return distance
