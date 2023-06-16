import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *


class Stimuli:

    def __init__(self):
        self.position = np.array([0, 0, 0])

    def draw_visualization(self, color=None):
        """
        Override this method to visualize the stimuli
        :return:
        """
        return None

    def set_position(self, position):
        self.position = position


class Cuboid(Stimuli):

    def __init__(self, a, b, c):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def get_force(self, position):
        """
        TODO
        :param position:
        :return:
        """


class Sphere(Stimuli):

    def __init__(self, radius):
        self.radius = radius
        super().__init__()

    def get_force(self, position):
        """
        Calculates the force on a node due to the stimulus
        :param position: position of the node (sensor)
        :return: force vector
        """
        x, y, z = position[0], position[1], position[2]

        # calculate the distance between the node and the center of the stimulus
        distance = np.sqrt(
            (x - self.position[0]) ** 2 + (y - self.position[1]) ** 2
        )
        # distance = np.sqrt(
        #     (x - self.position[0]) ** 2 + (y - self.position[1]) ** 2 + (z - self.position[2]) ** 2
        # )
        print('distance: ', distance)

        # check if the node is within the sphere of influence of the stimulus
        if distance < self.radius:
            # use a Gaussian-like distribution for the force, where the force is highest at the center of the
            # stimulus and decreases with distance Calculate the force on the node
            force_magnitude = np.exp(-distance**2 / (2 * self.radius**2))
            force_direction = (position - self.position) / distance  # unit vector pointing from stimulus_center to node
            return force_magnitude * force_direction
        else:
            return np.array([0.0, 0.0, 0.0])

    def draw_visualization(self, color=None):
        """
        :param color: it changes if the sphere is in use (being touched) on mouse click
        :return: an PyOpenGl sphere with given radius at given position
        """

        if color is None:
            color = [0.0, 0.0, 1.0, 1.0]

        # Load material
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.0, 0.0, 0.0, 1.0])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, color)
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])
        glMaterialfv(GL_FRONT, GL_SHININESS, 0.0)
        glMaterialfv(GL_FRONT, GL_EMISSION, [0.0, 0.0, 0.0, 1.0])

        # Create new sphere
        sphere = gluNewQuadric()
        gluQuadricDrawStyle(sphere, GLU_FILL)
        gluQuadricNormals(sphere, GLU_SMOOTH)
        gluQuadricOrientation(sphere, GLU_OUTSIDE)
        # slices: the number of subdivisions around the z-axis (similar to lines of longitude)
        # stacks: the number of subdivisions along the z-axis (similar to lines of latitude)
        gluQuadricTexture(sphere, GL_TRUE)
        gluSphere(sphere, self.radius, 16, 16)

    def set_position(self, position):
        self.position = np.copy(position)
