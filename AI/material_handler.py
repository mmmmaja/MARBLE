class Rank_Material:

    def __init__(self, density, young_modulus, poisson_ratio):
        """
        :param density: [g/cm^3]
            Measure of material's mass per unit volume

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

        self.density = density
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio

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

    def get_visual_properties(self):
        """
        TODO: add visualization properties
        :return: properties to be added to the visualizer
        """
        # Example settings for the metallic material
        color = 'linen'
        diffuse = 1
        specular = 0.5
        specular_power = 20
        metallic = 0.9
        roughness = 0.04


silicon = Rank_Material(
    density=2.329, young_modulus=140.0, poisson_ratio=0.265
)
rubber = Rank_Material(
    density=1.2, young_modulus=0.01, poisson_ratio=0.49
)

colors = [
            'purple',
            'violet'
            'magenta',
            'deepskyblue'
            'aqua'
            'turquoise'
            'lightgreen',
            'lime'
        ]
