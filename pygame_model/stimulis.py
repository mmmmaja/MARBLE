import numpy as np
from graphic_module import hex2RGB, Circle, Rectangle


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

    def get_shape(self, pos):
        return Rectangle(pos, self.a, self.b)


class Sphere(Stimuli):

    def __init__(self, radius):

        super().__init__()
        self.r = radius
        self.deform_r = 0

    def set_deformation(self, deform):
        super().set_deformation(deform)

        if abs(deform) < self.r:
            self.deform_r = np.sqrt(2 * deform * self.r - np.power(deform, 2))
        else:
            self.deform_r = self.r

    def get_distance(self, position):

        d_ = self.position - position

        return np.linalg.norm(d_) - self.deform_r

    def deformation_at(self, position):
        p_n = np.linalg.norm(self.position - position)
        if p_n > self.deform_r:
            return 0
        else:
            return np.sqrt(np.power(position, 2) + np.power(self.deform_r, 2))

    def get_shape(self, pos):
        return Circle(pos, self.r)
