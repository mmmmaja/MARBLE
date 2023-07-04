import pyvistaqt as pvqt  # For updating plots real time
from PyQt5.QtWidgets import QAction  # For the custom button

from AI.recording_manager import Recording


class GUI:

    def __init__(self, mesh_boost, mesh_material, stimuli, sensors):

        self.mesh_boost = mesh_boost
        # Rank material for rendering the mesh
        self.mesh_material = mesh_material
        # Stimuli object just for reference for the user
        self.stimuli = stimuli
        # List of sensors that record pressure
        self.sensors = sensors

        # Define the plotter (pyvistaqt)
        self.plotter = pvqt.BackgroundPlotter()

        # Define the pressure
        self.PRESSURE = 0.02
        # Change in the pressure when on the event
        self.pressure_dt = 0.05

        # Define all the actors present in the scene
        self.mesh_actor = None
        self.stimuli_actor = None
        self.sensor_actor = None
        self.material_text_actor = None
        self.mode_text_actor = None
        self.force_indicator_actor = None

        # Define the actions
        self.record_action = None
        self.stop_record_action = None
        # Recording object from recording_manager.py
        self.recording = None

        # Add all the actors to the scene
        self.draw_mesh()
        self.draw_stimuli()
        self.draw_sensors()
        self.add_material_text()
        self.add_mode_text('Interactive')
        self.add_axes()
        self.add_recording_actions()

        # Add the interactive events to the plotter
        self.add_force_events()
        self.add_pressure_indicator()

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

    def draw_mesh(self):
        """
        Adds the mesh in the .vtk format to the plotter
        """
        visual_properties = self.mesh_material.visual_properties
        self.mesh_actor = self.plotter.add_mesh(
            self.mesh_boost.current_vtk,
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

    def draw_sensors(self):
        # Add the point cloud to the plotter
        self.sensor_actor = self.plotter.add_points(
            self.sensors.visualization,
            render_points_as_spheres=True,
            color='#dfe9ff',
            point_size=6,
            name='sensor_points'
        )

    def add_mode_text(self, text):
        # Remove the text
        if self.mode_text_actor is not None:
            self.plotter.remove_actor(self.mode_text_actor)
        # Add the new text
        self.mode_text_actor = self.plotter.add_text(
            text, position='upper_right', font_size=10, color='white', shadow=True
        )

    def add_pressure_indicator(self):
        if self.force_indicator_actor is not None:
            self.plotter.remove_actor(self.force_indicator_actor)

        text = f'Pressure: {self.PRESSURE} N'
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
        self.PRESSURE = round(min(self.PRESSURE + self.pressure_dt, 5.0), 2)
        self.add_pressure_indicator()

    def decrease_force(self):
        self.PRESSURE = round(max(self.PRESSURE - self.pressure_dt, 0.0), 2)
        self.add_pressure_indicator()

    def add_force_events(self):

        # Add key event on the right arrow press
        self.plotter.add_key_event('Right', self.increase_force)

        # Add key event on the left arrow press
        self.plotter.add_key_event('Left', self.decrease_force)

    def add_recording_actions(self):
        self.record_action = QAction('Record', self.plotter.main_menu)
        self.record_action.triggered.connect(self.start_recording)
        self.plotter.main_menu.addAction(self.record_action)

        self.stop_record_action = QAction('Stop Recording', self.plotter.main_menu)
        self.stop_record_action.triggered.connect(self.stop_recording)
        self.stop_record_action.setVisible(False)  # initially hidden
        self.plotter.main_menu.addAction(self.stop_record_action)

    def start_recording(self):
        self.recording = Recording(self.sensors, file_name='test.csv')
        self.recording.start()
        self.update_recording_actions()

    def stop_recording(self):
        self.recording.stop()
        self.recording = None
        self.update_recording_actions()

    def update_recording_actions(self):
        if self.recording:
            self.record_action.setVisible(False)
            self.stop_record_action.setVisible(True)
        else:
            self.record_action.setVisible(True)
            self.stop_record_action.setVisible(False)
