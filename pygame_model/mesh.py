import sys

import numpy as np

# define unit = 1cm

UNIT = 40


class Mesh:

    def __init__(self, width, height, center=None, sensor_distance=1):
        """
        :param width: number of taxels horizontally
        :param height: number of taxels vertically
        """
        self.width = width
        self.height = height
        self.sensor_distance = sensor_distance

        if center:
            real_width, real_height = (width - 1) * (sensor_distance * UNIT), (height - 1) * (sensor_distance * UNIT)
            self.delta = [center[0] - real_width / 2, center[0] - real_width / 2]
        else:
            self.delta = [0, 0]

        self.SENSOR_ARRAY = []
        self.triangles = self.create()

    def create(self):
        """
        :return: Triangular mesh and point cloud corresponding to sensor positions
        """

        # Triangulation method
        vertices, triangles = [], []
        step = UNIT * self.sensor_distance

        for i in range(self.height - 1):
            for j in range(self.width - 1):
                a = [
                    i * step + self.delta[0],
                    j * step + self.delta[1]
                ]
                b = [
                    (i + 1) * step + self.delta[0],
                    j * step + self.delta[1]
                ]
                c = [
                    (i + 1) * step + self.delta[0],
                    (j + 1) * step + self.delta[1]
                ]
                d = [
                    i * step + self.delta[0],
                    (j + 1) * step + self.delta[1]
                ]

                self.SENSOR_ARRAY.append(Sensor(a))
                if i == self.height - 2:
                    self.SENSOR_ARRAY.append(Sensor(b))
                if j == self.width - 2:
                    self.SENSOR_ARRAY.append(Sensor(d))
                if i == self.height - 2 and j == self.width - 2:
                    self.SENSOR_ARRAY.append(Sensor(c))

                vertices.append(a)
                vertices.append(b)
                vertices.append(c)
                vertices.append(d)

                index = len(vertices) - 1
                triangles.append((vertices[index - 3], vertices[index - 2], vertices[index - 1]))
                triangles.append((vertices[index - 3], vertices[index - 1], vertices[index]))

        return triangles

    def get_values(self):
        for s in self.SENSOR_ARRAY:
            print(str(s.pressure), end=', ')

    def press(self, stimuli):
        for sensor in self.SENSOR_ARRAY:
            sensor.press(stimuli)


class Sensor:

    def __init__(self, position):
        self.deformation = 0
        self.position = np.array(position)
        self.activated = False

    def press(self, stimuli):

        distance = stimuli.get_distance(self.position)

        # stimuli directly presses on sensor
        if distance == 0:
            self.deformation = stimuli.deformation_at(self.position)
            self.activated = True

        # stimuli only deforms silicon where the sensor is on
        else:
            border_deformation = stimuli.border_deformation()

            self.deformation = stimuli.def_func.get_z(distance, border_deformation)
            self.activated = False

    def get_circle_properties(self):
        base_color = np.array([0, 0, 0])

        base_color[0] = min(255, int(base_color[0] - self.deformation * 10))
        base_color[1] = min(255, int(base_color[1] - self.deformation * 5))

        print(self.deformation)
        if self.activated:
            return [base_color, self.position, 6]
        else:
            return [base_color, self.position, 3]



"""
Stimula, specify radius/sides, start with c
function
distance to stimula
color code pressure
show deformation of chosen line
"""
