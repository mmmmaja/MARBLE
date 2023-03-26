import numpy as np
import csv

# define unit 40px = 1cm
UNIT = 40

# Here the pressure data will be saved
DATA = []


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

                self.SENSOR_ARRAY.append(Sensor(
                    frame_position=a,
                    real_position=np.array([a[0] - self.delta[0], a[1] - self.delta[1]]) / step
                ))

                vertices.append(a)
                vertices.append(b)
                vertices.append(c)
                vertices.append(d)

                index = len(vertices) - 1
                triangles.append((vertices[index - 3], vertices[index - 2], vertices[index - 1]))
                triangles.append((vertices[index - 3], vertices[index - 1], vertices[index]))

        print(self.SENSOR_ARRAY[0].real_position)
        print(self.SENSOR_ARRAY[1].real_position)
        return triangles

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
            data.append(i.deformation/UNIT)
        DATA.append(data)

    def save_data(self, path='data.csv'):
        sensor_positions = []
        for i in self.SENSOR_ARRAY:
            pos = str(i.real_position[0]) + ',' + str(i.real_position[1])
            sensor_positions.append(pos)
        DATA.insert(0, sensor_positions)
        with open(path, 'w', newline='') as file:
            print(path)
            writer = csv.writer(file)
            writer.writerows(DATA)


class Sensor:

    def __init__(self, frame_position, real_position=None):
        self.frame_position = np.array(frame_position)
        self.real_position = real_position
        self.activated = False
        self.deformation = 0

    def press(self, stimuli):

        distance = stimuli.get_distance(self.frame_position)

        # stimuli directly presses on sensor
        if distance == 0:
            self.deformation = stimuli.deformation_at(self.frame_position)
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
            return [base_color, self.frame_position, 6]
        else:
            return [base_color, self.frame_position, 3]





"""

1) step
The longer sensor is pressed the more pressure is observed

2)
transfer function: pressure to deformation
pressure vs softness

3) step
Relate transfer function to the program

for both:
think of bones, what shape? Softness of the tissue

check distribution for real data vs artificial one

Write script to obtain experiments
'forge a recording'
simulate running around with a stimuli
sliding



save first line true position of the sensor
"""
