import random

_colors = [
    '#e2a4ff',
    '#39ddff',
    '#63ffe1',
    '#ff9a9a'
]


def get_lighter_color(surface_color):
    """
    Returns a lighter version of the color
    :param surface_color: color of the surface
    :return: lighter color
    """
    # convert to rgb
    surface_color = surface_color.lstrip('#')
    surface_color = tuple(int(surface_color[i:i + 2], 16) for i in (0, 2, 4))
    # Get the lighter version of the color
    surface_color = tuple([min(255, x + 60) for x in surface_color])
    return surface_color


class Rank_Material:

    def __init__(self, name, young_modulus, poisson_ratio, time_constant, visual_properties=None):
        """

        :param young_modulus: [Gpa] (E)
             Property of the material that tells us how easily it can stretch and deform
             and is defined as the ratio of tensile stress (σ) to tensile strain (ε)

             A material with a high Young's modulus will be very stiff (like steel),
             while a material with a low Young's modulus will be very flexible (like rubber).

        :param poisson_ratio: [Gpa] (v)
            Poisson's ratio is a measure of the Poisson effect,
            the phenomenon in which a material tends to expand in directions perpendicular
            to the direction of compression.

            It is a property of the material that describes how a material tends to shrink
            in one direction when being stretched in another.
            For most materials, it's a value between 0 and 0.5.

        :param time_constant: [ms] (τ tau)
            The time constant is the time it takes for the material to reach 63.2% of its final value.
            Harder materials generally have shorter relaxation times than softer ones.
        """
        self.name = name
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.time_constant = time_constant

        if visual_properties is None:
            self.visual_properties = {
                'color': '#14ccff',
                'specular': 0.1,
                'metallic': 0.02,
                'roughness': 0.5,
                'edge_color': get_lighter_color('#14ccff'),
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
rubber_color = '2dc2ff'
steel_color = '787d91'
foam_color = '#a2a6f7'

# Create a database of materials

silicon = Rank_Material(
    name='Silicon',
    young_modulus=130.0, poisson_ratio=0.265,
    time_constant=1.0,
    visual_properties={
        'color': silicon_color,
        'specular': 0.1,
        'metallic': 0.02,
        'roughness': 0.5,
        'edge_color': get_lighter_color(silicon_color),
    }
)

rubber = Rank_Material(
    name='Rubber',
    young_modulus=0.05, poisson_ratio=0.19,
    time_constant=100.0,
    visual_properties={
        'color': rubber_color,
        'specular': 0.00,
        'metallic': 0.0,
        'roughness': 0.95,
        'edge_color': get_lighter_color(rubber_color),
    }
)

steel = Rank_Material(
    name='Steel',
    young_modulus=190.0, poisson_ratio=0.28,
    time_constant=1.0,
    visual_properties={
        'color': steel_color,
        'specular': 0.9,
        'metallic': 1.0,
        'roughness': 0.01,
        'edge_color': get_lighter_color(steel_color),
    }
)

polyurethane_foam = Rank_Material(
    name='Polyurethane foam',
    young_modulus=0.003, poisson_ratio=0.3,
    time_constant=1000.0,
    visual_properties={
        'color': foam_color,
        'specular': 0.0,
        'metallic': 0.0,
        'roughness': 1.0,
        'edge_color': get_lighter_color(foam_color),
    }
)
