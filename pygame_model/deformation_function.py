import numpy as np

class DeformationFunction:

    def __init__(self):

        pass

    def get_z(self, distance, object_deformation):
        if distance < 0:
            raise Exception(f"distance {distance} must be positive")

        return -np.exp((1/object_deformation)*distance - np.log(object_deformation))
