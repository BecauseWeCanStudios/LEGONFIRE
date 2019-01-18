import sys
import trimesh
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

def rotate_by_quaternion(v, q):
	w, q = q[0], q[1:]
	return 2 * np.dot(v, q) * q + (w ** 2 - np.dot(q, q)) * v + 2 * w * np.cross(q, v)

class LegoModelVisualizer:

	def __init__(self, models, data, colors=((0, 1, 0), (1, 0, 0)), width=800, height=600, verbose=False):
		self.models = [trimesh.load(i) for i in models]
		self.model_normals = [np.array(i.face_normals) for i in self.models]
		self.models = [np.array((i.triangles - i.center_mass)).reshape((-1, 3)) * 50 for i in self.models]
		self.data, self.colors, self.len, self.hidden = data, colors, len(data), [False] * len(colors)
		self.verbose = verbose

		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
		glutInitWindowSize(width, height)
		glutInit(sys.argv)
		glutCreateWindow(b'LEGO')

		glutDisplayFunc(self.get_draw_function())
		glutKeyboardFunc(self.get_special_keys_function())
		glutSpecialFunc(self.get_special_keys_function())

		glEnable(GL_CULL_FACE)
		glCullFace(GL_FRONT)
		glEnable(GL_DEPTH_TEST)
		glDisable(GL_BLEND)

		glEnable(GL_COLOR_MATERIAL)
		glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
		glMaterialfv(GL_FRONT, GL_SPECULAR, (1,1,1,1))
		glMaterialfv(GL_FRONT, GL_EMISSION, (0,0,0,1))

		glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.2, 0.2, 0.2 ,1))
		glEnable(GL_LIGHTING)
		glEnable(GL_LIGHT0)
		glLightfv(GL_LIGHT0, GL_POSITION, (0, 0, -1, 0))
		glLightfv(GL_LIGHT0, GL_AMBIENT, (0, 0, 0, 0))
		glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 0.5))
		glLightfv(GL_LIGHT0, GL_SPECULAR, (0.2, 0.2, 0.2, 0.2))
		glLightfv(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0)
		glLightfv(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0)
		glShadeModel(GL_SMOOTH)

		self.set_lego_model(0)
		glutMainLoop()

	def set_lego_model(self, index):
		self.index = index % self.len
		if self.verbose:
			print('Showing: ', self.index)
		model_id, self.offsets, self.rotations = self.data[self.index]
		model_id = int(model_id)
		normals = self.model_normals[model_id]
		self.current_model = self.models[model_id]
		self.vertices, self.normals = [], []
		for rot, off in zip(self.rotations, self.offsets):
			v = [None] * self.current_model.shape[0]
			for i in range(len(v)):
				v[i] = rotate_by_quaternion(self.current_model[i], rot) + [0, 0, 0.2]
			self.vertices.append(v)

			n = [None] * normals.shape[0]
			for i in range(len(n)):
				n[i] = rotate_by_quaternion(normals[i], rot)
			self.normals.append(n)

	def get_special_keys_function(self):

		def scpecial_keys(key, x, y):
			if key == GLUT_KEY_RIGHT:
				self.set_lego_model(self.index + 1)
			elif key == GLUT_KEY_LEFT:
				self.set_lego_model(self.index - 1)
			elif key in b'0123456789':
				index = int(key)
				if index < len(self.hidden):
					self.hidden[index] = not self.hidden[index]
			glutPostRedisplay()

		return scpecial_keys

	def get_draw_function(self):

		def draw():			
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glPushMatrix()

			glBegin(GL_TRIANGLES)
			for i in range(len(self.vertices)):
				if self.hidden[i]:
					continue
				verts = self.vertices[i]
				normals = self.normals[i]
				glColor3f(*self.colors[i])
				for j in range(0, len(verts), 3):
					for k in range(j, j + 3):
						glNormal3f(*normals[j // 3])
						glVertex3f(*verts[k])
			glEnd()

			glPopMatrix()
			glutSwapBuffers()

		return draw
