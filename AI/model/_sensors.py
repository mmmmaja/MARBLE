import csv
import sys
from abc import abstractmethod
import numpy as np
import pyvista

"""
This file contains the classes for the sensor arrays. 
We can create different sensor configurations for the mesh.
The sensors are used to measure the pressure at a certain point in the mesh.
"""


class SensorParent:

    # Minimal distance from the edges of the mesh
    offset = 1.4

    def __init__(self, mesh_boost):
        """
        Parent class for all the sensor configurations
        :param mesh_boost: The mesh object
        """

        self.mesh_boost = mesh_boost
        self.sensor_list = self.create_sensors()
        self.visualization = self.create_visualization()

    @abstractmethod
    def create_sensors(self) -> list:
        """
        Create the sensor list depending on the configuration
        :return: list of sensor objects
        """

    def map_vertex_ids(self, coordinates):
        """
        Map the coordinates to the closest vertex in the mesh
        :param coordinates: The coordinates of the sensor
        :return: The index of the closest vertex from the mesh
        """

        # Compute all the distances between the mesh points and the given coordinates
        distances = np.linalg.norm(self.mesh_boost.current_vtk.points - coordinates, axis=1)
        sensor_index = np.argmin(distances)
        return sensor_index

    def create_visualization(self):
        """
        Create the visualization of the sensors
        :return: PolyData object with the sensors that will be updated each time displacement happens
        """

        # Get the indices of the sensors
        sensor_indices = [sensor.index for sensor in self.sensor_list]
        # Get the coordinates of the sensors
        coords = self.mesh_boost.current_vtk.points[sensor_indices]
        # Create the PolyData object from the coordinates
        grid = pyvista.PolyData(coords)
        return grid

    def update_visualization(self):
        """
        Update the visualization of the sensors depending on the displacement of the mesh
        """

        # Get sensor positions
        sensor_indices = [sensor.index for sensor in self.sensor_list]
        coords = self.mesh_boost.current_vtk.points[sensor_indices]
        # Update the visualization
        self.visualization.points = coords

    def valid_distance(self, new_sensor, sensor_list, min_distance):
        """
        Check if the distance between the new sensor and all other sensors is bigger than min_distance
        :param new_sensor: The new sensor to be added to the list of valid
        :param sensor_list: List of all the sensors
        :param min_distance: The minimum distance between two sensors in the mesh
        :return: True if the distance is valid, False otherwise
        """

        for j in range(len(sensor_list)):
            j_sensor_coors = sensor_list[j].get_position(self.mesh_boost.current_vtk)
            _distance = np.linalg.norm(j_sensor_coors - new_sensor.get_position(self.mesh_boost.current_vtk))
            if _distance < min_distance:
                return False
        return True

    def relax(self):
        for sensor in self.sensor_list:
            sensor.stress = 0


class SensorGrid(SensorParent):

    def __init__(self, n_rows, n_cols, mesh_boost):
        """
        Used for grid-like meshes
        Create a grid of sensors
        :param n_rows: number of rows in the grid
        :param n_cols: number of columns in the grid
        :param mesh_boost: The mesh object
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        super().__init__(mesh_boost)

    def create_sensors(self):

        # Get the furthers points of the mesh
        bounds = self.mesh_boost.current_vtk.GetPoints().GetBounds()
        x_min, x_max = bounds[0] + self.offset, bounds[1] - self.offset
        y_min, y_max = bounds[2] + self.offset, bounds[3] - self.offset

        # Calculate the range of the mesh
        x_range, y_range = x_max - x_min, y_max - y_min

        # Fill the space with the sensors
        sensors = []
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                # Calculate the position of the sensor
                x = x_min + (x_range / (self.n_rows - 1)) * i
                y = y_min + (y_range / (self.n_cols - 1)) * j
                z = bounds[5]
                sensor_index = self.map_vertex_ids(np.array([x, y, z]))
                sensors.append(Sensor(name=f'{i}_{j}', index=sensor_index, mesh=self.mesh_boost))

        return sensors


class RandomSensors(SensorParent):

    def __init__(self, n_sensors, mesh_boost):
        """
        :param n_sensors: number of sensors in the list
        :param mesh_boost: The mesh object
        """
        self.n_sensors = n_sensors
        super().__init__(mesh_boost)

    def create_sensors(self):

        # Define minimal distance between two sensors
        min_distance = 0.5

        # Get the furthers points of the mesh
        bounds = self.mesh_boost.current_vtk.GetPoints().GetBounds()
        x_min, x_max = bounds[0] + self.offset, bounds[1] - self.offset
        y_min, y_max = bounds[2] + self.offset, bounds[3] - self.offset

        # Calculate the range of the mesh
        x_range, y_range = x_max - x_min, y_max - y_min

        # Indicates the number of iterations
        i = 0
        # Create list of sensors with random positions
        sensors = []
        while len(sensors) < self.n_sensors:
            # Create random position
            x = np.random.uniform(0, 1) * x_range + x_min
            y = np.random.uniform(0, 1) * y_range + y_min
            z = bounds[5]
            sensor_index = self.map_vertex_ids(np.array([x, y, z]))
            new_sensor = Sensor(name=f'{len(sensors)}', index=sensor_index, mesh=self.mesh_boost)
            if self.valid_distance(new_sensor, sensors, min_distance):
                sensors.append(new_sensor)
            i += 1
            if i > 2000:
                break

        return sensors


class SensorPatchesFromFile(SensorParent):

    def __init__(self, file_path, mesh_boost, n_patches=1):
        """
        Create sensors from a csv file
        :param file_path: path to the csv file with a sensor patch
        :param mesh_boost: The mesh object
        :param n_patches: number of patches to create (The patch from the file will be repeated n_patches times)
        """

        self.file_path = file_path
        self.n_patches = n_patches
        super().__init__(mesh_boost)

    def create_sensors(self):
        # Read the csv file
        """
        The csv file should have the following format:
        | x1 | y1 | -> sensor 1 coordinates
        | x2 | y2 | -> (..)
        | x3 | y3 |
        :return: list of sensors
        """
        # Open the file and read the initial coordinates of the sensors
        initial_coords = []
        with open(self.file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            for row in csv_reader:
                # | x | y | position of the sensor
                initial_coords.append(np.array([
                    float(row[0]), float(row[1])
                ]))

        print(f'File {self.file_path} has been read')

        # Compute the ranges of the coordinates in the file
        x_min_file = min([coords[0] for coords in initial_coords])
        x_max_file = max([coords[0] for coords in initial_coords])
        x_range_file = x_max_file - x_min_file

        y_min_file = min([coords[1] for coords in initial_coords])
        y_max_file = max([coords[1] for coords in initial_coords])
        y_range_file = y_max_file - y_min_file

        # Add new patches of sensors to the list

        # Compute the translation of the initial patch in the 2D space
        max_range = max(x_range_file, y_range_file)
        # Vector by which the sensors will be translated in the 2D space
        translation = np.array([max_range, max_range])

        # Extend the list by adding new patches of sensors
        extended_coords = initial_coords.copy()
        for i in range(1, self.n_patches):
            for j in range(len(initial_coords)):
                sensor = initial_coords[j].copy()
                sensor += translation * i
                extended_coords.append(sensor)

        # Get the bounds of the mesh the sensors are going to be placed on
        bounds = self.mesh_boost.current_vtk.GetPoints().GetBounds()
        x_min, x_max = bounds[0] + self.offset, bounds[1] - self.offset
        y_min, y_max = bounds[2] + self.offset, bounds[3] - self.offset
        # Calculate the range of the mesh
        x_range, y_range = x_max - x_min, y_max - y_min

        # Recalculate the ranges of the coordinates in the file once again when new sensors are added (extended_coords)
        x_min_file = min([coords[0] for coords in extended_coords])
        x_max_file = max([coords[0] for coords in extended_coords])
        x_range_file = x_max_file - x_min_file

        y_min_file = min([coords[1] for coords in extended_coords])
        y_max_file = max([coords[1] for coords in extended_coords])
        y_range_file = y_max_file - y_min_file

        # Scale the sensors positions to the mesh
        sensors = []
        for i in range(len(extended_coords)):
            x_scaled = (extended_coords[i][0] - x_min_file) / x_range_file * x_range + x_min
            y_scaled = (extended_coords[i][1] - y_min_file) / y_range_file * y_range + y_min
            z = bounds[5]
            sensor_index = self.map_vertex_ids(np.array([x_scaled, y_scaled, z]))
            sensors.append(Sensor(name=f'{i}', index=sensor_index, mesh=self.mesh_boost))

        return sensors


class SensorArm(SensorParent):

    def __init__(self, mesh_boost):
        """
        Used for arm mesh
        The sensors are placed on the arm and are saved in a .obj file
        :param mesh_boost: The mesh object
        """
        super().__init__(mesh_boost)

    def create_sensors(self):
        sensors = []
        # Load the input points from the .obj file
        points = pyvista.read('../meshes/sensors.obj').points
        for i in range(points.shape[0]):
            # Map the points to the closest vertex in the mesh
            sensor_index = self.map_vertex_ids(points[i])
            # Create the sensor object
            sensors.append(Sensor(name=f'{i}', index=sensor_index, mesh=self.mesh_boost))

        return sensors


class Sensor:

    def __init__(self, name, index, mesh):
        """
        :param name: Unique name of the sensor
        :param index: Index of the sensor in the mesh

        Mimics SingleTact sensor, the area of contact is approximately 8 mm^2
        """
        self.name = name
        self.index = index
        self.stress = 0.0

        self.initial_position = self.get_position(mesh.current_vtk)
        self.neighbour_cells = mesh.get_neighbouring_cells(index)

    def get_position(self, vtk_mesh):
        return vtk_mesh.points[self.index]

    def get_stress(self):
        """
        Stress is a measure of the internal forces in a material body.
        It's defined as the force per unit area.
        """
        return self.stress

    def set_readings(self, stress_output):
        """
        In a 3D stress tensor,
        the normal stresses are the diagonal elements, namely σ_xx, σ_yy, and σ_zz.
        """

        average_reading = 0.0
        n = len(self.neighbour_cells)
        for i in range(n):
            average_reading += stress_output[self.neighbour_cells[i]][2]
        average_reading /= n
        self.stress = average_reading


"""
Patch Creator functions to write csv files with different shape of patches
"""


def circle(n_points, n_loops, radius):
    """
    Creates a circle patch with the given parameters
    :param n_points: Number of points in each loop
    :param n_loops: Number of loops of the circle
    :param radius: Radius of the furthest loop
    """

    with open('../patches/circle.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        min_radius = 0.1
        radius_range = radius - min_radius
        for j in range(n_loops):
            radius = min_radius + radius_range * j / n_loops
            for i in range(n_points):
                angle = 2 * np.pi * i / n_points
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                writer.writerow([x, y])


def spiral(n_points):
    """
    Creates a spiral patch with the given parameters
    :param n_points: Number of points in the spiral
    """

    with open('../patches/spiral.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        for i in range(n_points):
            angle = 0.1 * i
            radius = angle
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            writer.writerow([x, y])


def star(n_points, n_arms):
    """
    Creates a star patch with the given parameters
    :param n_points: Total number of points in the star
    :param n_arms: Number of arms of the star
    """
    with open('../patches/star.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            radius = np.abs(np.sin(n_arms * angle))
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            writer.writerow([x, y])


