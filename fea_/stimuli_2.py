import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *


class Stimuli:

    def __init__(self):
        self.position = np.array([0, 0, 0])

    def set_position(self, position):
        self.position = np.copy(position)

    def draw_visualization(self):
        """
        Override this method to visualize the stimuli
        :return:
        """
        return None

    def use(self):
        """
        Override this method to use the stimuli
        :return:
        """
        return None


class Cuboid(Stimuli):

    def __init__(self, a, b, c):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def get_distance(self, position):
        """
        TODO
        :param position:
        :return:
        """


class Sphere(Stimuli):

    def __init__(self, radius):
        self.radius = radius
        super().__init__()

    def get_distance(self, position):
        """
        TODO
        :param position:
        :return:
        """

    def draw_visualization(self, in_use=False):
        """
        :param in_use: if the sphere is in use (being touched) on mouse click
        :return: an PyOpenGl sphere with given radius at given position
        """

        # Create new sphere
        self.load_material(in_use)
        sphere = gluNewQuadric()
        gluQuadricDrawStyle(sphere, GLU_FILL)
        gluQuadricNormals(sphere, GLU_SMOOTH)
        gluQuadricOrientation(sphere, GLU_OUTSIDE)
        # slices: the number of subdivisions around the z-axis (similar to lines of longitude)
        # stacks: the number of subdivisions along the z-axis (similar to lines of latitude)
        gluQuadricTexture(sphere, GL_TRUE)
        gluSphere(sphere, self.radius, 32, 32)

    def load_material(self, in_use):
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.0, 0.0, 0.0, 1.0])
        if in_use:
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.0, 1.0, 0.0, 1.0])
        else:
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.0, 0.0, 1.0, 1.0])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])
        glMaterialfv(GL_FRONT, GL_SHININESS, 0.0)
        glMaterialfv(GL_FRONT, GL_EMISSION, [0.0, 0.0, 0.0, 1.0])
