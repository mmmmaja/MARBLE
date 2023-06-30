import numpy as np
from graphic_module import *

OFFSET = np.array([70, 70, 0])


class Stimuli:

    def __init__(self, def_func):
        self.def_func = def_func

        self.deformation = 0
        self.position = np.array([0, 0, 0])

    def set_deformation(self, deform):
        self.deformation = deform

    def set_frame_position(self, position):
        position = (position - OFFSET) / UNIT
        self.position = np.copy(position)
        print("new position: ", position)


class Cuboid(Stimuli):

    def __init__(self, def_func, a, b):

        super().__init__(def_func)
        self.a = a
        self.b = b

    def get_distance(self, position):

        d_ = position - self.position

        if abs(d_[0]) <= self.a / 2 and abs(d_[1]) <= self.a / 2:
            return 0
        elif abs(d_[0]) <= self.a / 2:
            return abs(d_[1]) - self.b / 2
        elif abs(d_[1]) <= self.b / 2:
            return abs(d_[0]) - self.a / 2
        else:
            return np.linalg.norm(np.absolute(d_) - np.array([self.a / 2, self.b / 2,0]))

    def deformation_at(self, position):

        if self.get_distance(position) == 0:
            return self.deformation
        else:
            return 0

    def border_deformation(self):
        return self.deformation

    def get_shape(self):
        return Rectangle((self.position * UNIT + OFFSET).astype(int), int(self.a * UNIT), int(self.b * UNIT))


class Sphere(Stimuli):

    def __init__(self, def_func, radius):

        super().__init__(def_func)
        self.r = radius
        self.deform_r = 0
        self.border_deform = 0

    def set_deformation(self, deform):
        super().set_deformation(deform)

        deform_ = deform * 0.5
        if abs(deform_) < self.r:
            self.deform_r = np.sqrt(2 * abs(deform_) * self.r - np.power(deform_, 2))
        else:
            self.deform_r = self.r

        self.border_deform = deform_

    def get_distance(self, position):

        d_ = np.linalg.norm(self.position - position)
        if d_ <= self.deform_r:
            return 0
        else:
            return d_ - self.deform_r

    def deformation_at(self, position):
        p_n = np.linalg.norm(self.position - position)
        if p_n > self.deform_r:
            return 0
        else:
            return \
                -np.sqrt(-np.sum(np.power(self.position - position, 2)) +
                         np.power(self.r, 2)) + (self.deformation + self.r)

    def border_deformation(self):
        return self.border_deform

    def get_shape(self):
        return Circle((self.position*UNIT + OFFSET).astype(int), int(self.deform_r*UNIT))
