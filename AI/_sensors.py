from abc import abstractmethod
import numpy as np
import pyvista

"""
This file contains the classes for the sensors. 
The sensors are used to measure the pressure at a certain point in the mesh.
"""


class SensorParent:

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

        # Distance from the edge of the mesh
        offset = 1.4
        x_min, x_max = bounds[0] + offset, bounds[1] - offset
        y_min, y_max = bounds[2] + offset, bounds[3] - offset

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
                sensors.append(Sensor(name=f'{i}_{j}', index=sensor_index))

        return sensors


class SensorArm(SensorParent):

    def __init__(self, mesh_boost):
        """
        Used for arm meshes
        The sensors are placed on the arm and are saved in a .obj file
        :param mesh_boost: The mesh object
        """
        super().__init__(mesh_boost)

    def create_sensors(self):
        sensors = []
        # Load the input points from the .obj file
        points = pyvista.read('meshes/sensors.obj').points
        for i in range(points.shape[0]):
            # Map the points to the closest vertex in the mesh
            sensor_index = self.map_vertex_ids(points[i])
            # Create the sensor object
            sensors.append(Sensor(name=f'{i}', index=sensor_index))

        return sensors


class Sensor:

    def __init__(self, name, index):
        """
        :param name: Unique name of the sensor
        :param index: Index of the sensor in the mesh
        """
        self.name = name
        self.index = index
        self.pressure = 0.0

    def get_position(self, vtk_mesh):
        return vtk_mesh.points[self.index]
