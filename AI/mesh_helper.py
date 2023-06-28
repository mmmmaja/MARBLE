import math
import pyvista


PATH = 'meshes/mesh.vtk'


def display_mesh(mesh_path=PATH):
    """
    Displays the mesh with .vtk extension
    :param mesh_path: path to the mesh
    """

    # Check if the mesh is a path or a meshio mesh
    try:
        mesh = pyvista.read(mesh_path)
    except FileNotFoundError:
        print("Mesh not found")
        return

    mesh.plot(
        show_edges=True,
        color='deepskyblue',
        show_scalar_bar=False,
    )


def concave(i, j, width, height):
    # Make the convexity dependent on the width and height of the mesh
    # The bump is in the middle of the mesh
    bump_factor = 0.05
    centre = [width / 2, height / 2]
    x = i - centre[0]
    y = j - centre[1]
    z = - bump_factor * (x ** 2 + y ** 2)
    return z


def flat(i, j, width, height):
    return 0.0


def convex(i, j, width, height):
    # Make the convexity dependent on the width and height of the mesh
    # The bump is in the middle of the mesh
    bump_factor = 0.05
    centre = [width / 2, height / 2]
    x = i - centre[0]
    y = j - centre[1]
    z = bump_factor * (x ** 2 + y ** 2)
    return z


def wave(i, j, width, height):
    # Make the wave independent on the width and height of the mesh
    bump_factor = 0.55
    x = i
    y = j
    z = bump_factor * math.sin(x) * math.sin(y)
    return z