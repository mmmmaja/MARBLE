import numpy as np


class Stimuli:

    def __init__(self):
        self.deformation = 0
        self.position = np.array([0, 0])

    def set_deformation(self, deform):
        self.deformation = deform

    def set_position(self, position):
        self.position = np.copy(position)


class Cuboid(Stimuli):

    def __init__(self, a, b):

        super().__init__()
        self.a = a
        self.b = b

    def get_distance(self, position):

        d_ = position - self.position

        if abs(d_[0]) <= self.a / 2 and abs(d_[1]) <= self.a / 2:
            return 0
        elif abs(d_[0]) <= self.a / 2:
            return abs(d_[1]) - self.b / 2
        elif abs(d_[1]) <= self.b / 2:
            return abs(d_[1]) - self.a / 2
        else:
            return np.linalg.norm(np.absolute(d_) - np.array([self.a / 2, self.b / 2]))

    def deformation_at(self, position):

        if self.get_distance(position) == 0:
            return self.deformation
        else:
            return 0


class Sphere(Stimuli):

    def __init__(self, radius):
        super().__init__()
        self.r = radius

    def get_distance(self, position):
        d_ = self.position - position

        return
