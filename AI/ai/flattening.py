import igl
import numpy as np
from matplotlib import pyplot as plt


# https://libigl.github.io/libigl-python-bindings/tut-chapter4/

def harmonic_parametrization(vertices, triangles):
    # find the boundary of the mesh
    bnd = igl.boundary_loop(triangles)

    # map the boundary to a circle
    bnd_uv = igl.map_vertices_to_circle(vertices, bnd)

    # compute the harmonic parameterization
    uv = igl.harmonic_weights(vertices, triangles, bnd, bnd_uv, 1)

    return uv


def as_rigid_as_possible(vertices, triangles):
    """
    As-rigid-as-possible surface parameterization
    preserve distances (so angles as well)
    """

    # Find the open boundary
    bnd = igl.boundary_loop(triangles)

    # Map the boundary to a circle, preserving edge proportions
    bnd_uv = igl.map_vertices_to_circle(vertices, bnd)

    # Harmonic parametrization for the internal vertices
    uv = igl.harmonic_weights(vertices, triangles, bnd, bnd_uv, 1)

    arap = igl.ARAP(vertices, triangles, 2, np.zeros(0))
    uva = arap.solve(np.zeros((0, 0)), uv)

    plot_mesh_2D(uva, triangles)


def least_squares_conformal_maps(vertices, triangles):
    """
    Least squares conformal maps aim to preserve the angles between neighboring triangles.
    It does not need to have a fixed boundary.
    """

    # Find the boundary of the mesh
    boundary = igl.boundary_loop(triangles)

    # Fix two arbitrary points on the boundary
    b = np.array([2, 1])
    b[0] = boundary[0]
    b[1] = boundary[int(boundary.size / 2)]

    bc = np.array([[0.0, 0.0], [1.0, 0.0]])

    # LSCM parametrization
    _, uv = igl.lscm(vertices, triangles, b, bc)

    return uv


def plot_mesh_2D(uv, triangles):
    plt.figure()

    # add edges
    for tri in triangles:
        i, j, k = tri
        # Use magenta thin lines for the edges
        plt.plot([uv[i, 0], uv[j, 0]], [uv[i, 1], uv[j, 1]], color='m', linewidth=0.5)
        plt.plot([uv[j, 0], uv[k, 0]], [uv[j, 1], uv[k, 1]], color='m', linewidth=0.5)
        plt.plot([uv[k, 0], uv[i, 0]], [uv[k, 1], uv[i, 1]], color='m', linewidth=0.5)

    # scatter plot of vertices
    plt.scatter(uv[:, 0], uv[:, 1])

    plt.show()


def plot_mesh_3D(vertices, triangles):
    # Make the interactive 3D plot with matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', marker='o')

    # Plot the triangles
    for tri in triangles:
        i, j, k = tri
        # Plot a triangle
        ax.plot([vertices[i, 0], vertices[j, 0]], [vertices[i, 1], vertices[j, 1]],
                [vertices[i, 2], vertices[j, 2]], 'm')
        ax.plot([vertices[j, 0], vertices[k, 0]], [vertices[j, 1], vertices[k, 1]],
                [vertices[j, 2], vertices[k, 2]], 'm')
        ax.plot([vertices[k, 0], vertices[i, 0]], [vertices[k, 1], vertices[i, 1]],
                [vertices[k, 2], vertices[i, 2]], 'm')

    plt.show()


def get_mesh_descriptor(mesh):
    vertices = np.array([sensor.position for sensor in mesh.SENSOR_ARRAY])
    triangles = np.array(mesh.delaunay_points)
    return vertices, triangles


def surface_flattening(mesh, mapping=least_squares_conformal_maps):
    """
    Surface Parameterization
    Finds a mapping Ψ from the 3D surface to a 2D plane that best preserves the geometric properties of the 3D surface.

    :param mesh: triangular mesh object with sensor points and delaunay simplices
    :param mapping: type of mapping to be used
    :return list uv, consisting of 2D points of sensors
    """

    vertices, triangles = get_mesh_descriptor(mesh)
    plot_mesh_3D(vertices, triangles)
    uv = mapping(vertices, triangles)
    plot_mesh_2D(uv, triangles)
    return uv


"""
Types of mapping:

as_rigid_as_possible

harmonic_parametrization
    Maps boundaries to a circular shape

least_squares_conformal_maps

"""
