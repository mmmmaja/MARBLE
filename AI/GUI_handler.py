import pyvistaqt as pvqt  # For updating plots real time


class GUI:

    def __init__(self, mesh_material, vtk_mesh):

        # Rank material for rendering the mesh
        self.mesh_material = mesh_material

        # Define the plotter (pyvistaqt)
        self.plotter = pvqt.BackgroundPlotter()

        # Define all the actors present in the scene
        self.mesh_actor = None
        self.material_text_actor = None
        self.mode_text_actor = None

        # Add all the actors to the scene
        self.draw_mesh(vtk_mesh)
        self.add_material_text()
        self.add_mode_text('Interactive')

        # Update the plotter and show it
        self.plotter.update()
        self.plotter.show()

    def add_material_text(self):
        # Add the material name to the plotter
        text = self.mesh_material.name
        text += '\nE: ' + str(self.mesh_material.young_modulus)
        text += '\nv: ' + str(self.mesh_material.poisson_ratio)

        self.material_text_actor = self.plotter.add_text(
            text, position='lower_left', font_size=8, color='white', shadow=True
        )

    def draw_mesh(self, vtk_mesh):
        """
        :param vtk_mesh: mesh in .vtk format (instance of mesh_boost)

        Took it out of the Main to create a separate timer class
        Adds the mesh in the .vtk format to the plotter
        """
        if self.mesh_actor is not None:
            self.plotter.remove_actor(self.mesh_actor)

        visual_properties = self.mesh_material.visual_properties
        self.mesh_actor = self.plotter.add_mesh(
            vtk_mesh,
            show_edges=True,
            smooth_shading=True,
            show_scalar_bar=False,
            edge_color=visual_properties['edge_color'],
            color=visual_properties['color'],
            specular=visual_properties['specular'],
            metallic=visual_properties['metallic'],
            roughness=visual_properties['roughness'],
            name='initial_mesh'
        )

    def add_mode_text(self, text):
        # Remove the text
        if self.mode_text_actor is not None:
            self.plotter.remove_actor(self.mode_text_actor)
        # Add the new text
        self.mode_text_actor = self.plotter.add_text(
            text, position='upper_right', font_size=8, color='white', shadow=True
        )
