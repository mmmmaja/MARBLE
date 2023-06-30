import math
import numpy as np
from matplotlib import pyplot as plt
from AI.ai.flattening import surface_flattening


def encloses(triangle, point):
    """
    Check if a point lays within a 2D triangle
    :param triangle: Consists of 3 points in 2D space
    :param point: Point in 2D space
    :return: True if point lays within triangle, False otherwise
    """

    # Use baycentric coordinates to check if point lays within triangle

    # Any point P within the triangle can be expressed as:
    # P = A + u * (C - A) + v * (B - A)

    # Where A, B, C are the vertices of the triangle
    # u, v are the barycentric coordinates of P
    # Solve for u and v

    # Get the vectors of the triangle
    v0 = triangle[2] - triangle[0]
    v1 = triangle[1] - triangle[0]
    v2 = point - triangle[0]

    u = (np.dot(v1, v1) * np.dot(v2, v0) - np.dot(v1, v0) * np.dot(v2, v1)) \
        / (np.dot(v0, v0) * np.dot(v1, v1) - np.dot(v0, v1) * np.dot(v1, v0))

    v = (np.dot(v0, v0) * np.dot(v2, v1) - np.dot(v0, v1) * np.dot(v2, v0)) \
        / (np.dot(v0, v0) * np.dot(v1, v1) - np.dot(v0, v1) * np.dot(v1, v0))

    # Check if point lays within triangle
    if v < 0 or u < 0:
        return False
    if u + v > 1:
        return False
    return True


def get_enclosing_triangle(mesh, uv, resolution, x_rc):
    """
    Find the triangle in which a given point lays
    :param mesh: mesh of the sensor array
    :param uv: 2D mapping of the mesh
    :param resolution: image resolution
    :param x_rc: point in 2D space that we need to find the enclosing triangle for
    :return: position of the triangle, pressure of the triangle
    """

    # Find triangle in which given point lays
    for triangle in mesh.delaunay_points:

        # Get real position of the triangle with respect to 2D mapping and resolution
        position = []
        for index in triangle:
            position.append(uv[index] * resolution)

        # Find out if x_rc lays within this triangle
        if encloses(position, x_rc):
            # Find the corresponding pressure
            pressure = [mesh.SENSOR_ARRAY[index].pressure for index in triangle]
            return position, pressure

    # No triangle found
    return None, None


def get_maximum_pressure(mesh):
    # Returns maximum pressure observed in the mesh
    return max(
        [sensor.pressure for sensor in mesh.SENSOR_ARRAY]
    )


def get_pixel_color(mesh, p_rc):
    P = get_maximum_pressure(mesh)
    if P == 0:
        return 0
    else:
        return math.floor((255 / P) * p_rc)


def get_area(p1, p2, p3):
    # Compute the area of a triangle in 2D
    return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))


def get_pressure(mesh, uv, resolution, x_rc):
    """
    Compute the pressure in p[x(r,c)] using the barycentric interpolation
    :param mesh: mesh of the sensor array
    :param uv: 2D mapping of the mesh
    :param resolution: image resolution
    :param x_rc: point in 2D space that we need to find the pressure for
    :return: pressure in p[x(r,c)]
    """

    # Find the enclosing triangle of x(r,c)
    # Consists of 3 mapped points onto 2D surface
    triangle_positions, triangle_pressure = get_enclosing_triangle(mesh, uv, resolution, x_rc)
    if triangle_positions is None:
        return 0.0

    # The positions of the triangle points
    m_j, m_k, m_h = triangle_positions

    # The pressure registered in the corresponding sensors
    p_j, p_k, p_h = triangle_pressure

    # get the areas of the triangles
    A1 = get_area(m_j, m_k, x_rc)
    A2 = get_area(m_k, x_rc, m_h)
    A3 = get_area(m_h, m_j, x_rc)
    A = get_area(m_j, m_k, m_h)

    # Compute the pressure in p[x(r,c)] using the barycentric interpolation
    return (A1 * p_h + A2 * p_k * A3 * p_j) / A


def show_image(image, uv=None, triangles=None, resolution=None):
    if uv:
        # add edges
        for tri in triangles:
            i, j, k = tri
            # Use magenta thin lines for the edges

            plt.plot(
                [uv[i, 0] * resolution, uv[j, 0] * resolution], [uv[i, 1] * resolution, uv[j, 1] * resolution],
                color='m', linewidth=0.5
            )
            plt.plot(
                [uv[j, 0] * resolution, uv[k, 0] * resolution], [uv[j, 1] * resolution, uv[k, 1] * resolution],
                color='m', linewidth=0.5
            )
            plt.plot(
                [uv[k, 0] * resolution, uv[i, 0] * resolution], [uv[k, 1] * resolution, uv[i, 1] * resolution],
                color='m', linewidth=0.5
            )

        # scatter plot of vertices
        plt.scatter(uv[:, 0] * resolution, uv[:, 1] * resolution)

    # # show image
    # # Move the origin of the axes to the center of the array
    # plt.imshow(image, cmap='gray', extent=[0, uv[:, 0].max() * resolution, 0, uv[:, 1].max() * resolution])
    plt.show()


def form_contact_image(mesh):
    # Get 2D mapping of a 3D mesh (2D numpy array)
    uv = surface_flattening(mesh)

    # Create a pressure map
    # Create a grid for the image
    resolution = 100
    x_max, y_max = max(uv[:, 0]), max(uv[:, 1])

    R = int(y_max * resolution)  # number of rows in a grid
    C = int(x_max * resolution)  # number of columns in a grid

    # y_max -> y_max * resolution
    # x_max -> x_max * resolution

    # x_i -> x_i * resolution

    image = np.zeros((R, C))
    print("Image resolution: ", R, C)

    # Compute the pressure in p[x(r,c)] using the barycentric interpolation
    for r in range(R):
        for c in range(C):
            x_rc = np.array([c, r])
            p_rc = get_pressure(mesh, uv, resolution, x_rc)
            k_rc = get_pixel_color(mesh, p_rc)
            image[r, c] = k_rc

    # Map the grid onto the image

    show_image(uv, mesh.delaunay_points, image, resolution)
    # show_image(image)
    return image
