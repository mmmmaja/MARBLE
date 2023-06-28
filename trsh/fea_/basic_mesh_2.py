import numpy as np
import csv
from scipy.spatial import Delaunay

# Here the pressure data will be saved
# FIXME it's so ugly
DATA = []


def triangulate(sensor_array):
    """
    Triangulation of the 3D surface
    Creates a triangulation of the sensor array in order to display mesh for the points
    """

    # First save all the (x,y,z) positions of the sensors
    positions = []
    for sensor in sensor_array:
        positions.append(sensor.position)
    # n x 3 array of points
    positions = np.array(positions)

    # Triangulate projections
    delaunay = Delaunay(positions[:, :2])
    # Consists of indices of nodes [p1, p2, p3]
    return delaunay.simplices


class Mesh:

    def __init__(self):
        self.SENSOR_ARRAY = self.create()
        self.delaunay_points = triangulate(self.SENSOR_ARRAY)
        self.give_IDs()

    def give_IDs(self):
        # Give IDs to the sensors in the sensor array
        # Used for Finite Element method later on
        for i in range(len(self.SENSOR_ARRAY)):
            self.SENSOR_ARRAY[i].ID = i

    def create(self):
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
        sum_x = sum(point.position[0] for point in self.SENSOR_ARRAY)
        sum_y = sum(point.position[1] for point in self.SENSOR_ARRAY)
        centroid_x = sum_x / total_points
        centroid_y = sum_y / total_points
        return [centroid_x, centroid_y]

    def update_geometry(self, u):
        # Update the geometry of the mesh
        # u is the displacement vector
        for i in range(len(u)):
            u_i = u[i]
            self.SENSOR_ARRAY[i].update_geometry(u_i)


class Sensor:

    def __init__(self, position):

        self.position = np.array(position)

        self.activated = False
        self.deformation = 0
        self.pressure = 0

        self.ID = None

    def press(self, stimuli):
        distance = stimuli.get_distance(self.position)

    def update_geometry(self, u):
        # Update the geometry of the sensor
        self.position += u
