import numpy as np

class DeformationFunction:

    def __init__(self):

        pass

    def get_z(self, distance, deformation_level):
        deformation_level = abs(deformation_level)
        if distance < 0:
            raise Exception(f"distance {distance} must be positive")

        return -np.exp(-( (1/deformation_level)*distance - np.log(deformation_level)) )
