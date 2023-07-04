from abc import abstractmethod

import numpy as np
import pyvista


class SensorParent:

    def __init__(self, mesh_boost):
        self.mesh_boost = mesh_boost
        self.sensor_list = self.create()
        self.visualization = self.create_visualization()

    @abstractmethod
    def create(self):
        pass

    def map_vertex_ids(self, coordinates):
        # Find the closest vertex in the mesh
        distances = np.linalg.norm(self.mesh_boost.current_vtk.points - coordinates, axis=1)
        sensor_index = np.argmin(distances)
        return sensor_index

    def create_visualization(self):
        # get the visualization of the points
        sensor_indices = [sensor.index for sensor in self.sensor_list]
        coords = self.mesh_boost.current_vtk.points[sensor_indices]
        # Create an unstructured grid with the points
        grid = pyvista.PolyData(coords)
        return grid

    def update_visualization(self):
        # get the visualization of the points
        sensor_indices = [sensor.index for sensor in self.sensor_list]
        coords = self.mesh_boost.current_vtk.points[sensor_indices]
        # Create an unstructured grid with the points
        self.visualization.points = coords


class SensorGrid(SensorParent):

    def __init__(self, n_rows, n_cols, mesh_boost):

        self.n_rows = n_rows
        self.n_cols = n_cols
        super().__init__(mesh_boost)

    def create(self):
        # Create a point cloud with the sensors
        # Get the furthers points of the mesh
        bounds = self.mesh_boost.current_vtk.GetPoints().GetBounds()

        offset = 1.4
        x_min, x_max = bounds[0] + offset, bounds[1] - offset
        y_min, y_max = bounds[2] + offset, bounds[3] - offset

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

        super().__init__(mesh_boost)

    def create(self):
        sensors = []
        # Read the .obj opath and convert it to a vtk mesh
        # Load the input points
        points = pyvista.read('meshes/sensors.obj').points
        for i in range(points.shape[0]):
            sensor_index = self.map_vertex_ids(points[i])
            sensors.append(Sensor(name=f'{i}', index=sensor_index))

        return sensors


class Sensor:
    def __init__(self, name, index):
        self.name = name
        self.index = index
        self.pressure = 0.0

    def get_position(self, vtk_mesh):
        return vtk_mesh.points[self.index]
