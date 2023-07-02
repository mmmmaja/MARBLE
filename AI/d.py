import pyvista as pv

# Create a sphere
sphere = pv.Sphere()

# Create the plotter with two renderers
plotter = pv.Plotter(shape=(1, 2))  # 1 row, 2 columns

# First renderer
plotter.subplot(0, 0)  # Select the first renderer
plotter.add_mesh(sphere, color="blue")

# Create a second sphere at the same position
sphere2 = pv.Sphere(center=sphere.center)

# Second renderer
plotter.subplot(0, 1)  # Select the second renderer
plotter.add_mesh(sphere2, color="red")

# Show the plotter
plotter.show()
