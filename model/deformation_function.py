import numpy as np


class DeformationFunction:

    def __init__(self):
        pass

    def get_z(self, distance, deformation_level, softness=1):

        deformation_level = abs(deformation_level)
        if deformation_level == 0:
            return 0
        if distance < 0:
            raise Exception(f"distance {distance} must be positive")

        return -np.exp(-((softness / deformation_level) * distance - np.log(deformation_level)))
