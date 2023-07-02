import pyvistaqt as pvqt  # For updating plots real time


class GUI:

    def __init__(self, vtk_mesh, mesh_material, stimuli):

        # Rank material for rendering the mesh
        self.mesh_material = mesh_material

        self.stimuli = stimuli

        # Define the plotter (pyvistaqt)
        self.plotter = pvqt.BackgroundPlotter()

        self.FORCE = 0.02
        self.force_dt = 0.01

        # Define all the actors present in the scene
        self.mesh_actor = None
        self.stimuli_actor = None
        self.material_text_actor = None
        self.mode_text_actor = None
        self.force_indicator_actor = None

        # Add all the actors to the scene
        self.draw_mesh(vtk_mesh)
        self.draw_stimuli()
        self.add_material_text()
        self.add_mode_text('Interactive')
        self.add_axes()

        self.add_force_events()
        self.add_force_indicator()

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
            self.plotter.update()
            # self.plotter.remove_actor(self.mesh_actor)

        visual_properties = self.mesh_material.visual_properties
        self.mesh_actor = self.plotter.add_mesh(
            vtk_mesh,
            show_edges=False,
            smooth_shading=True,
            show_scalar_bar=False,
            edge_color=visual_properties['edge_color'],
            color=visual_properties['color'],
            specular=visual_properties['specular'],
            metallic=visual_properties['metallic'],
            roughness=visual_properties['roughness'],
            name='initial_mesh'
        )

    def draw_stimuli(self):
        if self.stimuli_actor is not None:
            self.plotter.update()

        self.stimuli_actor = self.plotter.add_mesh(
            self.stimuli.create_visualization(),
            color=self.stimuli.color,
            name='stimuli',
            show_edges=False,
            smooth_shading=True,
            specular=0.8,
            metallic=0.95,
            roughness=0.0,
        )

    def add_mode_text(self, text):
        # Remove the text
        if self.mode_text_actor is not None:
            self.plotter.remove_actor(self.mode_text_actor)
        # Add the new text
        self.mode_text_actor = self.plotter.add_text(
            text, position='upper_right', font_size=8, color='white', shadow=True
        )

    def add_force_indicator(self):
        if self.force_indicator_actor is not None:
            self.plotter.remove_actor(self.force_indicator_actor)

        text = f'Force: {self.FORCE} N'
        self.force_indicator_actor = self.plotter.add_text(
            text, position='lower_right', font_size=8, color='white', shadow=True
        )

    def add_axes(self):
        self.plotter.add_axes(
            line_width=3, viewport=(0, 0.1, 0.2, 0.3),
            x_color='08a9ff', y_color='#FF00FF',
            z_color='#00ff8d'
        )

    def increase_force(self):
        self.FORCE = round(min(self.FORCE + self.force_dt, 5.0), 2)
        self.add_force_indicator()

    def decrease_force(self):
        self.FORCE = round(max(self.FORCE - self.force_dt, 0.0), 2)
        self.add_force_indicator()

    def add_force_events(self):

        # Add key event on the right arrow press
        self.plotter.add_key_event('Right', self.increase_force)

        # Add key event on the left arrow press
        self.plotter.add_key_event('Left', self.decrease_force)
