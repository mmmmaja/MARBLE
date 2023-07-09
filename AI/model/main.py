import vtk
from PyQt5.QtWidgets import QApplication

from AI.model import pressure_script
from AI.model.GUI_handler import GUI
from AI.model._sensors import SensorPatchesFromFile, SensorGrid
from AI.model.fenics import *
from AI.model.mesh_converter import *
from AI.model.material_handler import *
from AI.model.stimulis import *
import io
import sys
from AI.model.pressure_script import *
from AI.model._activation import ActivationClass


# Set to True to enable the terminal output,
# otherwise the output will be redirected to the log file (maybe it is faster this way)
TERMINAL_OUTPUT = True


class Main:

    def __init__(self, mesh_boost, stimuli, sensors, rank_material):
        self.stimuli = stimuli
        self.sensors = sensors
        self.fenics = FENICS(mesh_boost, rank_material, sensors)

        self.gui = GUI(mesh_boost, rank_material, stimuli, sensors)
        self.add_interactive_events()

    def add_interactive_events(self):
        # Enable cell picking
        self.gui.plotter.enable_cell_picking(
            # if cell is not none, apply force to the cell
            callback=lambda cell: apply_cell_specific_pressure(self.fenics, self.gui, cell),
            font_size=10, point_size=30, line_width=2,
            color='white', style='wireframe', through=False
        )

        # Add the event on the press of the space bar, apply the force
        self.gui.plotter.add_key_event(
            'space', lambda: apply_volume_pressure(self.fenics, self.gui)
        )

        # If the enter button is pressed, the interactive mode is toggled
        self.gui.plotter.add_key_event('Control_L', self.toggle_interactive)
        self.gui.plotter.show()

    def toggle_interactive(self):
        print("Interactive mode toggled")

        if self.gui.plotter.interactor.GetInteractorStyle().__class__ == ActivationClass:
            self.gui.plotter.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            self.gui.add_mode_text("Interactive")
        else:
            self.gui.add_mode_text("Activation")
            self.gui.plotter.interactor.SetInteractorStyle(ActivationClass(
                gui=self.gui,
                fenics=self.fenics,
                stimuli=self.stimuli
            ))


if not TERMINAL_OUTPUT:
    # create a text trap and redirect stdout
    text_trap = io.StringIO()
    sys.stdout = text_trap

app = QApplication(sys.argv)

_mesh_boost = GridMesh(30, 30, z_function=flat, layers=3)
_sensors = SensorGrid(12, 12, _mesh_boost)
# _sensors = RandomSensors(20, _mesh_boost)

# _mesh_boost = ArmMesh()
# _sensors = SensorArm(_mesh_boost)
# _sensors = SensorPatchesFromFile("../patches/circle.csv", _mesh_boost, n_patches=4)

# _stimuli = Cylinder(radius=3.0, height=1.0)
_stimuli = Cuboid(6.0, 4.0, 2.0)
# _stimuli = Sphere(radius=2.1)


# force_handler = pressure_script.StimuliPressure(_stimuli, 10, rubber)
# FENICS(_mesh_boost, rubber, _sensors).apply_pressure(force_handler)


Main(_mesh_boost, _stimuli, _sensors, rubber)
app.exec_()


"""
TODO:
Stress relaxation process
Apply pressure to sensors
Add README
Add random mesh grid
Add maximum displacement (look at the fucking foam at 8N)

Very important!!!
Check why displacement is 0 when there is contact!!!
"""
