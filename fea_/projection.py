import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

def normalize(vector):
    return vector / np.linalg.norm(vector)


def calculate_ray(mouse_pos, SCREEN_WIDTH, SCREEN_HEIGHT):
    """
    Model View Projection is a common series of matrix transformations that can be applied to a vertex
    defined in model space, transforming it into clip space, which can then be rasterized.

    Model matrix
    Projection matrix
    View matrix
    v' = P * M * V * v
    """

    # transform pos into 3d normalised device coordinates
    # From -1 to 1
    x = (2.0 * mouse_pos[0]) / SCREEN_WIDTH - 1.0
    y = 1.0 - (2.0 * mouse_pos[1]) / SCREEN_HEIGHT

    # If you choose 1, you'll get a point on the far clip plane.
    # If you choose 0, you'll get a point on the near clip plane
    z = 1.0
    # Represent it as a 4D vector
    ray_nds = np.array([x, y, z, 1.0])


def get_3D_coordinates(pos):
    x, y = pos

    viewport = glGetIntegerv(GL_VIEWPORT)
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)

    winX = float(x)
    winY = float(viewport[3] - y)  # subtract y from viewport height to flip the Y axis

    # get the z value of the clicked pixel
    winZ = glReadPixels(x, int(winY), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)

    # unproject
    posX, posY, posZ = gluUnProject(winX, winY, winZ, modelview, projection, viewport)

    return posX, posY, posZ
