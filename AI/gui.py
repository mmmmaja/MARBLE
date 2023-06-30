import sys

import pyvistaqt as pvqt  # For updating plots real time
import vtk


class GUIHandler:

    def __init__(self, mesh_boost, rank_material, stimuli):
        self.mesh_boost = mesh_boost
        self.rank_material = rank_material
        self.stimuli = stimuli
        self.plotter = None

        # Define the actors
        self.mode_text_actor = None
        self.material_text_actor = None
        self.mesh_actor = None
        self.stimuli_actor = None

        self.mesh_mapper = vtk.vtkDataSetMapper()
        self.mesh_mapper.SetInputData(self.mesh_boost.current_vtk)

        self.create_plot()

    def create_plot(self):
        self.plotter = pvqt.BackgroundPlotter()

        self.draw_mesh()
        self.draw_stimuli()
        self.add_material_text()
        self.add_mode_text()

    def add_material_text(self):
        text = self.rank_material.name
        text += '\nE: ' + str(self.rank_material.young_modulus)
        text += '\nv: ' + str(self.rank_material.poisson_ratio)
        self.material_text_actor = self.plotter.add_text(
            text, position='lower_left', font_size=8, color='white', shadow=True
        )

    def draw_mesh(self):
        if self.mesh_actor is not None:
            self.plotter.remove_actor(self.mesh_actor)
        _visual_properties = self.rank_material.visual_properties
        self.mesh_actor = self.plotter.add_mesh(
            self.mesh_boost.current_vtk,
            show_edges=True,
            smooth_shading=True,
            show_scalar_bar=False,
            edge_color=_visual_properties['edge_color'],
            color=_visual_properties['color'],
            specular=_visual_properties['specular'],
            metallic=_visual_properties['metallic'],
            roughness=_visual_properties['roughness'],
            name='mesh'
        )
        self.mesh_actor.SetMapper(self.mesh_mapper)

        self.plotter.enable_lightkit()
        self.plotter.update()

    def draw_stimuli(self):
        if self.stimuli_actor is not None:
            self.plotter.remove_actor(self.stimuli_actor)

        self.stimuli_actor = self.plotter.add_mesh(
            self.stimuli.get_visualization(),
            show_edges=False,
            smooth_shading=True,
            color='magenta',
            name='stimuli'
        )

    def add_mode_text(self):
        self.mode_text_actor = self.plotter.add_text(
            "Interactive", position='upper_right', font_size=8, color='white', shadow=True
        )

    def update_mode_text(self, text):
        # Remove the text
        self.plotter.remove_actor(self.mode_text_actor)
        # Add the new text
        self.mode_text_actor = self.plotter.add_text(
            text, position='upper_right', font_size=8, color='white', shadow=True
        )
        self.plotter.update()

