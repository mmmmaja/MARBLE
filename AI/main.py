import vtk
from PyQt5.QtWidgets import QApplication
from AI.GUI_handler import GUI
from AI.fenics import *
from AI.mesh_converter import *
from AI.material_handler import *
from AI.stimulis import *
import io
import sys
from AI.pressure_script import *
from AI._activation import ActivationClass


# Set to True to enable the terminal output,
# otherwise the output will be redirected to the log file (maybe it is faster this way)
TERMINAL_OUTPUT = True


class Main:

    def __init__(self, mesh_boost, stimuli, sensors, rank_material):
        self.stimuli = stimuli
        self.sensors = sensors
        self.fenics = FENICS(mesh_boost, rank_material)

        self.gui = GUI(mesh_boost, rank_material, stimuli, sensors)
        self.add_interactive_events()

    def add_interactive_events(self):
        # Enable cell picking
        self.gui.plotter.enable_cell_picking(
            # if cell is not none, apply force to the cell
            callback=lambda cell: apply_cell_specific_pressure(self.fenics, self.gui, cell),
            font_size=10, point_size=30, line_width=5,
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
_sensors = SensorGrid(10, 10, _mesh_boost)

# _mesh_boost = ArmMesh()
# _sensors = SensorArm(_mesh_boost)

_stimuli = Sphere(radius=2.1)
# _stimuli = Cylinder(radius=3.0, height=1.0)
# _stimuli = Cuboid(6.0, 4.0, 2.0)


Main(_mesh_boost, _stimuli, _sensors, rubber)
app.exec_()


"""
TODO:
Stress relaxation process
Apply pressure to sensors
Add README
"""
