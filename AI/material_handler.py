import random

_colors = [
    '#ad1a95',
    '#FF00FF',
    'e874ff',
    '#e2a4ff',
    '#689dff',
    '#14ccff',
    '#17d8db',
    '#62fff8',
    '#98ff9a',
    '#ff9a9a'
]


def get_edge_color(surface_color):
    """
    Returns a lighter version of the color
    :param surface_color: color of the surface
    :return: lighter color
    """
    # convert to rgb
    surface_color = surface_color.lstrip('#')
    surface_color = tuple(int(surface_color[i:i + 2], 16) for i in (0, 2, 4))
    # Get the lighter version of the color
    surface_color = tuple([int((x + 255) / 2) for x in surface_color])
    return surface_color


class Rank_Material:

    def __init__(self, young_modulus, poisson_ratio, visual_properties=None):
        """

        :param young_modulus: [Gpa]
             Property of the material that tells us how easily it can stretch and deform
             and is defined as the ratio of tensile stress (σ) to tensile strain (ε)

             A material with a high Young's modulus will be very stiff (like steel),
             while a material with a low Young's modulus will be very flexible (like rubber).

        :param poisson_ratio: [Gpa]
            Poisson's ratio is a measure of the Poisson effect,
            the phenomenon in which a material tends to expand in directions perpendicular
            to the direction of compression.

            It is a property of the material that describes how a material tends to shrink
            in one direction when being stretched in another.
            For most materials, it's a value between 0 and 0.5.
        """

        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio

        if visual_properties is None:
            self.visual_properties = {
                'color': 'linen',
                'diffuse': 1,
                'specular': 0.5,
                'specular_power': 20,
                'metallic': 0.9,
                'roughness': 0.04
            }
        else:
            self.visual_properties = visual_properties

    def get_properties(self):
        """
        Returns additional material properties
        :return: mu and lambda
        """
        E = self.young_modulus
        nu = self.poisson_ratio

        # Mu is the shear modulus (shows the material's resistance to deformation)
        mu = E / (2.0 * (1.0 + nu))
        # Lambda is the Lame parameter (defines the relationship between stress and strain)
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # Return the properties in the correct format for the SfePy solver
        return {
            'lam': lam,
            'mu': mu,
            'alpha': 0.00  # thermal expansion coefficient
        }


silicon_color = random.choice(_colors)
rubber_color = '08a9ff'
steel_color = 'C0C0C0'
foam_color = '#ffc291'

# Create a database of materials

silicon = Rank_Material(
    young_modulus=140.0, poisson_ratio=0.265,
    visual_properties={
        'color': silicon_color,
        'specular': 0.1,
        'metallic': 0.02,
        'roughness': 0.5,
        'edge_color': get_edge_color(silicon_color),
    }
)

rubber = Rank_Material(
    young_modulus=0.05, poisson_ratio=0.49,
    visual_properties={
        'color': rubber_color,
        'specular': 0.00,
        'metallic': 0.0,
        'roughness': 0.95,
        'edge_color': get_edge_color(rubber_color),
    }
)

steel = Rank_Material(
    young_modulus=190.0, poisson_ratio=0.28,
    visual_properties={
        'color': steel_color,
        'specular': 0.9,
        'metallic': 1.0,
        'roughness': 0.01,
        'edge_color': get_edge_color(steel_color),
    }
)

polyurethane_foam = Rank_Material(
    young_modulus=0.003, poisson_ratio=0.3,
    visual_properties={
        'color': foam_color,
        'specular': 0.0,
        'metallic': 0.0,
        'roughness': 1.0,
        'edge_color': get_edge_color(foam_color),
    }
)
# F(t) = F0 * exp(-t/tau)
# TODO add time step Tau
