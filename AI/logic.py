from PyQt5.QtWidgets import QApplication
from AI.GUI_handler import GUI
from AI.fenics import FENICS
from AI.mesh_converter import GridMesh
from AI.material_handler import rubber
from AI.stimulis import Sphere
from AI.stress_script import StressRelaxation


class SimulationLogic:

    TERMINAL_OUTPUT = True
    stress_relaxation_ref = None
    FORCE_LIM = 0.2

    def __init__(self, mesh_boost, stimuli, rank_material):
        self.stimuli = stimuli
        self.fenics = FENICS(mesh_boost, rank_material)


    def toggle_interactive(self):
        ...

    def apply_stimuli(self, fenics, stimuli):
        """Applies stimuli based on a provided force limit"""
        force = self._get_forces_from_stimuli(fenics, stimuli)
        return force

    def _get_forces_from_stimuli(self, fenics, stimuli):
        ...
        return force

    ...


