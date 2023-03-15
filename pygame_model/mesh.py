import math
import pygame

DISTANCE_BETWEEN_SENSORS = 40
# define unit = 1cm


class Mesh:

    def __init__(self, width, height, center=None):
        """
        :param width: number of taxels horizontally
        :param height: number of taxels vertically
        """
        self.width = width
        self.height = height

        if center:
            real_width, real_height = (width - 1) * DISTANCE_BETWEEN_SENSORS, (height - 1) * DISTANCE_BETWEEN_SENSORS
            self.delta = [center[0] - real_width / 2, center[0] - real_width / 2]
        else:
            self.delta = [0, 0]

        self.SENSOR_ARRAY = []

        self.vertices, self.triangles = self.create()

    def create(self):
        """
        :return: Triangular mesh and point cloud corresponding to sensor positions
        """

        # FIXME unit for the visualization

        # Triangulation method
        vertices, triangles = [], []
        step = DISTANCE_BETWEEN_SENSORS

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

                vertices.append(a)
                vertices.append(b)
                vertices.append(c)
                vertices.append(d)

                index = len(vertices) - 1
                triangles.append((vertices[index - 3], vertices[index - 2], vertices[index - 1]))
                triangles.append((vertices[index - 3], vertices[index - 1], vertices[index]))

        return vertices, triangles

    def get_values(self):
        for s in self.SENSOR_ARRAY:
            print(str(s.pressure), end=', ')

    def press(self, pos):
        for sensor in self.SENSOR_ARRAY:
            sensor.press(pos)


class Sensor:

    def __init__(self, position):
        self.pressure = 0
        self.position = position
        self.activated = False

    def get_distance(self, point):
        # FIXME
        return math.sqrt(
            math.pow(self.position[0] - point[0], 2) + math.pow(self.position[1] - point[1], 2)
        )

    def press(self, pos):
        self.pressure = 1 / self.get_distance(pos)
        if self.pressure > 0.02:
            self.activated = True
        else:
            self.activated = False


"""
Stimula, specify radius/sides, start with cube
function
distance to stimula
color code pressure
show deformation of chosen line
"""
