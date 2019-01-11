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
	for i in (-0.0187293, 0.0742610, -0.2121144, 1.5707288):
		ret += i
		ret *= x
	ret *= K.sqrt(1 - x)
	ret -= 2 * negate * ret
	return negate * pi + ret

def MeshMetric(models):
	meshes = [trimesh.load(i) for i in models]
	for mesh in meshes:
		mesh.vertices = (mesh.vertices - mesh.center_mass) * 50
	lens = [len(i.vertices) for i in meshes]
	n = max(lens) + 1
	meshes = K.constant(np.stack([np.concatenate([np.array(i.vertices), np.zeros((n - j, 3))], axis=0) for i, j in zip(meshes, lens)]))
	mask = K.constant([[j < l for j in range(n)] for l in lens], dtype='float32')

	def metric(y_true, y_pred):
		y_true, classes = y_true[..., :-1], K.cast(K.round(y_true[..., -1]), 'int32')
		p_true, o_true, p_pred, o_pred = K.expand_dims(y_true[..., :3], 1), y_true[..., 3:], K.expand_dims(y_pred[..., :3], 1), y_pred[..., 3:]
		m = K.gather(meshes, classes)
		pred = rotate_vector_by_quaternion(o_pred, m)
		true = rotate_vector_by_quaternion(o_true, m)
		diff = K.sum(K.pow(pred + p_pred - true - p_true, 2), axis=-1)
		diff = K.maximum(-y_pred[:, 0], K.sum(diff * K.gather(mask, classes), axis=-1))
		return diff

	return metric

def QuaternionDistanceMetric():

	def metric(y_true, y_pred):
		return K.clip(1 - K.square(y_true[..., 3:-1] * y_pred[..., 3:]), 0, 1)

	return metric

def QuaternionAngleMetric():

	def metric(y_true, y_pred):
		return acos(K.clip(2 * K.square(y_true[..., 3:-1] * y_pred[..., 3:]) - 1, -1, 1)) / pi * 180

	return metric
