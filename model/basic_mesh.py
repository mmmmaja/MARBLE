from _csv import reader
import numpy as np
import csv
from scipy.spatial import Delaunay
from model.graphic_module import UNIT

# Here the pressure data will be saved
DATA = []


def set_frame_position(sensor_array):

    # Check if coordinates are negative and if yes, then shift the whole frame
    min_x, min_y = 0, 0
    for sensor in sensor_array:
        if sensor.real_position[0] < min_x:
            min_x = sensor.real_position[0]
        if sensor.real_position[1] < min_y:
            min_y = sensor.real_position[1]

    # Shift mesh from the corners to the center
    delta = [50, 50]
    for sensor in sensor_array:
        sensor.real_position = sensor.real_position - [min_x, min_y, 0]
        sensor.frame_position = sensor.real_position[:2] * UNIT + delta


def triangulate(sensor_array):
    # Creates a triangulation of the sensor array in order to display mesh for the points
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
    return delaunay.simplices, triangles


class Mesh:

    def __init__(self):
        self.SENSOR_ARRAY = self.create()
        set_frame_position(self.SENSOR_ARRAY)
        self.delaunay_points, self.triangles = triangulate(self.SENSOR_ARRAY)
        self.displayed_points = self.set_displayed_points()
        self.give_IDs()

    def give_IDs(self):
        # Give IDs to the sensors in the sensor array
        for i in range(len(self.SENSOR_ARRAY)):
            self.SENSOR_ARRAY[i].ID = i

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

    def get_central_point(self):
        total_points = len(self.SENSOR_ARRAY)
        sum_x = sum(point.real_position[0] for point in self.SENSOR_ARRAY)
        sum_y = sum(point.real_position[1] for point in self.SENSOR_ARRAY)
        centroid_x = sum_x / total_points
        centroid_y = sum_y / total_points
        return [centroid_x, centroid_y]


class Sensor:

    def __init__(self, real_position=None):

        self.real_position = np.array(real_position)
        self.frame_position = None

        self.activated = False
        self.deformation = 0

        self.ID = None

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
