import numpy as np


class Stimuli:

    def __init__(self):
        self.position = np.array([0, 0, 0])

    def set_position(self, position):
        self.position = np.copy(position)


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

        super().__init__()
        self.radius = radius

    def get_distance(self, position):
        """
        TODO
        :param position:
        :return:
        """
