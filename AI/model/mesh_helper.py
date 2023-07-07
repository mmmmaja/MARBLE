import math
import random

"""
This file has helper functions for the mesh generation.
The functions are used to generate the height of the mesh at a given point.
"""


def flat(i, j, width, height):
    """
    :param i: current x position
    :param j: current y position
    :param width: width of the mesh
    :param height: height of the mesh
    :return: the height of the mesh at a given point (i, j)
    """

    return 0.0


def concave(i, j, width, height):
    # The bump is in the middle of the mesh
    bump_factor = 0.05
    centre = [width / 2, height / 2]
    x = i - centre[0]
    y = j - centre[1]
    z = - bump_factor * (x ** 2 + y ** 2)
    return z


def convex(i, j, width, height):
    # The concavity is in the middle of the mesh
    bump_factor = 0.05
    centre = [width / 2, height / 2]
    x = i - centre[0]
    y = j - centre[1]
    z = bump_factor * (x ** 2 + y ** 2)
    return z


def wave(i, j, width, height):
    # Specify how much the wave should be scaled
    wave_factor = 0.75
    z = wave_factor * math.sin(i) * math.sin(j)
    return z


def cylinder(i, j, width, height):
    # Simulate the heightmap of a cylinder

    # Cylinder parameters
    radius = min(width, height) / 2
    center = [width / 2, height / 2]

    # Distance from center
    x = i - center[0]
    y = j - center[1]
    dist = math.sqrt(x ** 2 + y ** 2)

    # Calculate height
    if dist <= radius:
        z = math.sqrt(radius ** 2 - dist ** 2)
    else:
        z = 0

    return z


def random_noise(i, j, width, height):
    # Create terrain with random variations

    # Random noise parameters
    max_height = 2.0

    # Generate height
    z = random.uniform(0, max_height)

    return z

