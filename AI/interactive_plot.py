import time
from random import random, randint

import numpy as np
import pyvista as pv
from pyvista import _vtk

#TODO pyvista.Plotter.track_click_position


def define_random_displacements(mesh):
    u = np.zeros_like(mesh.points)
    random_range = 0.1
    # Fill displacements with random values
    x = np.random.rand(mesh.n_points) * random_range
    y = np.random.rand(mesh.n_points) * random_range
    z = np.random.rand(mesh.n_points) * random_range
    u[:, 0] = x
    u[:, 1] = y
    u[:, 2] = z
    return u


class InteractivePlot:

    def __init__(self, mesh):
        self.mesh = mesh
        self.plotter = pv.Plotter()
        self.actor = None
        self.create()
        # self.update()

    def create(self):

        # Add the mesh
        self.actor = self.plotter.add_mesh(
            self.mesh,
            show_edges=True,
            color='aqua',
            show_scalar_bar=False,
            smooth_shading=True,
            lighting=False
        )
        # Set up the mouse callback
        # self.plotter.iren.add_observer('LeftButtonPressEvent', self.mouse_callback)
        self.plotter.show(auto_close=False, interactive_update=True)
        self.plotter.track_click_position(callback=self.mouse_callback, side='right', double=False, viewport=False)

    def mouse_callback(self, obj, event):
        print("Mouse event: ", event)

    def update(self):
        # Decide on your update frequency (for instance, 20 times per second)
        FREQUENCY = 20.0
        update_interval = 1.0 / FREQUENCY

        try:
            while True:  # Or some condition to stop the loop when necessary
                start_time = time.time()

                displacements = define_random_displacements(self.mesh)
                self.mesh.points = self.mesh.points + displacements
                self.plotter.update_coordinates(self.mesh.points)

                # Render the updated mesh
                self.plotter.render()

                # Calculate the time taken to calculate the displacements and render the mesh
                elapsed_time = time.time() - start_time

                # If updating and rendering took less time than your desired update interval,
                # sleep for the remaining time
                if elapsed_time < update_interval:
                    time.sleep(update_interval - elapsed_time)

        except KeyboardInterrupt:
            # Allow the program to exit on a KeyboardInterrupt (Ctrl+C)
            pass



