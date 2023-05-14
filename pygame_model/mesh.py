from _csv import reader
import numpy as np
import csv
from scipy.spatial import Delaunay
from graphic_module import UNIT

# Here the pressure data will be saved
DATA = []


def set_frame_position(sensor_array):
    delta = [50, 50]
    for sensor in sensor_array:
        sensor.frame_position = sensor.real_position[:2] * UNIT + delta


def triangulate(sensor_array):
    positions = []
    for s in sensor_array:
        positions.append(s.frame_position)

    delaunay = Delaunay(np.array(positions))  # triangulate projections
    triangles = []
    for triangle in delaunay.simplices:
        t = []
        for i in triangle:
            t.append(positions[i])
        triangles.append(t)
    return triangles


class Mesh:

    def __init__(self):
        self.SENSOR_ARRAY = self.create()
        set_frame_position(self.SENSOR_ARRAY)
        self.triangles = triangulate(self.SENSOR_ARRAY)
        self.displayed_points = self.set_displayed_points()

    def create(self):
        # To be overridden by a child class
        return None

    def set_displayed_points(self):
        # To be overridden by a child class
        return None

    def press(self, stimuli):
        # Record the pressure
        for sensor in self.SENSOR_ARRAY:
            sensor.press(stimuli)

    def append_data(self):
        data = []
        for i in self.SENSOR_ARRAY:
            data.append(i.deformation)
        DATA.append(data)

    def save_data(self, path='fake_data/data.csv'):

        sensor_positions = []
        for i in self.SENSOR_ARRAY:
            pos = str(i.real_position[0]) + ',' + str(i.real_position[1]) + ',' + str(i.real_position[2])
            sensor_positions.append(pos)
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(sensor_positions)
            writer.writerows(DATA)


class Sensor:

    def __init__(self, real_position=None):

        self.real_position = np.array(real_position)
        self.frame_position = None

        self.activated = False
        self.deformation = 0

    def press(self, stimuli):
        distance = stimuli.get_distance(self.real_position)

        # stimuli directly presses on sensor
        if distance == 0:
            self.deformation = stimuli.deformation_at(self.real_position)
            self.activated = True

        # stimuli only deforms silicon where the sensor is on
        else:
            border_deformation = stimuli.border_deformation()

            self.deformation = stimuli.def_func.get_z(distance, border_deformation)
            self.activated = False

    def get_circle_properties(self):
        base_color = np.array([0, 0, 0])
        base_color[2] = min(255, int(base_color[2] - self.deformation * UNIT * 10))
        base_color[1] = min(255, int(base_color[1] - self.deformation * UNIT * 5))

        if self.activated:
            return [base_color, self.frame_position, 6]
        else:
            return [base_color, self.frame_position, 3]
