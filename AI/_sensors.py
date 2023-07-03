import numpy as np
import pyvista


class SensorGrid:

    def __init__(self, n_rows, n_cols, mesh_boost):

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.mesh_boost = mesh_boost

        self.sensors = self.create()

    def get_vertex_ids(self, coordinates):
        # Find the closest vertex in the mesh
        distances = np.linalg.norm(self.mesh_boost.current_vtk.points - coordinates, axis=1)
        sensor_index = np.argmin(distances)
        return sensor_index

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
                sensor_index = self.get_vertex_ids(np.array([x, y, z]))
                sensors.append(Sensor(name=f'{i}_{j}', index=sensor_index))

        return sensors

    def get_visualization(self):
        # get the visualization of the points
        sensor_indices = [sensor.index for sensor in self.sensors]
        return self.mesh_boost.current_vtk.points[sensor_indices]


class SensorArm:
    def __init__(self, path='meshes/sensors.obj'):
        self.vtk = self.convert_to_vtk(path)

    def convert_to_vtk(self, path):
        # Read the .obj opath and convert it to a vtk mesh
        # Load the input points
        return pyvista.read(path)

    def get_visualization(self):
        # get the visualization of the points
        # https://docs.pyvista.org/version/stable/examples/02-plot/point-clouds.html
        return self.vtk.points


class Sensor:
    def __init__(self, name, index):
        self.name = name
        self.index = index
        self.value = 0.0

    def update(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def get_name(self):
        return self.name
