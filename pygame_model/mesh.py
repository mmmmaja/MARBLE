import numpy as np
import csv

# define unit 40px = 1cm
UNIT = 40

# Here the pressure data will be saved
DATA = []


def save_data():
    with open('data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(DATA)


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
            # Used to shift mesh to the center of the frame
            real_width, real_height = (width - 1) * (sensor_distance * UNIT), (height - 1) * (sensor_distance * UNIT)
            self.delta = [center[0] - real_width / 2, center[0] - real_width / 2]
        else:
            self.delta = [0, 0]

        self.SENSOR_ARRAY = []
        self.triangles = self.create()

    def create(self):
        """
        Fills self.SENSOR_ARRAY
        :return: Triangular mesh for sensor map
        """

        # Triangulation method
        vertices, triangles = [], []
        step = UNIT * self.sensor_distance

        for i in range(self.height):
            for j in range(self.width):
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

        return triangles

    def get_values(self):
        # print pressure data on key press
        for i in range(self.height):
            for j in range(self.width):
                index = i * self.width + j
                print(round(self.SENSOR_ARRAY[index].deformation, 5), end=' | ')
            print()

    def press(self, stimuli):
        # Record the pressure
        for sensor in self.SENSOR_ARRAY:
            sensor.press(stimuli)

    def get_points_along_X(self, X):
        sensor_line = []
        for i in range(self.height):
            index = i * self.width + X
            sensor_line.append(self.SENSOR_ARRAY[index])
        return sensor_line

    def append_data(self):
        data = []
        for i in self.SENSOR_ARRAY:
            data.append(i.deformation)
        DATA.append(data)


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

        base_color[2] = min(255, int(base_color[2] - self.deformation * 10))
        base_color[1] = min(255, int(base_color[1] - self.deformation * 5))

        if self.activated:
            return [base_color, self.position, 6]
        else:
            return [base_color, self.position, 3]





"""
The longer sensor is pressed the more pressure is observed
show deformation of chosen line
"""
