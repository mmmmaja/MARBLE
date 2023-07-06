import vtk
from AI.pressure_script import *


class ActivationClass(vtk.vtkInteractorStyleTrackballCamera):
    """
    This class was added to override the default mouse events of the vtkInteractorStyleTrackballCamera class.
    It disables the rotation of the mesh when the left mouse button is pressed.
    Triggered in Activation mode
    """

    def __init__(self, parent=None, gui=None, fenics=None, stimuli=None, relaxation=None, *args, **kwargs):

        self.gui = gui
        self.fenics = fenics
        self.stimuli = stimuli

        # Mouse pressed flag for mesh activation
        self.mouse_pressed = False

        # Create a cell picker for the mesh (to map the mouse coordinates to the mesh)
        self.picker = vtk.vtkCellPicker()
        self.picker.AddPickList(self.gui.mesh_actor)

        # Override the default mouse events (disable rotation)
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("MiddleButtonPressEvent", self.middle_button_press_event)
        self.AddObserver("RightButtonPressEvent", self.right_button_press_event)
        self.AddObserver("LeftButtonReleaseEvent", self.left_button_release_event)
        self.AddObserver("MouseMoveEvent", self.mouse_move_event)

        super().__init__(*args, **kwargs)

    def left_button_press_event(self, obj, event):
        self.mouse_pressed = True

    def left_button_release_event(self, obj, event):
        self.mouse_pressed = False

    def middle_button_press_event(self, obj, event):
        # Disable the middle button events
        pass

    def right_button_press_event(self, obj, event):
        # Disable the right button events
        pass

    def mouse_move_event(self, obj, event):
        """
        Function that is triggered when the mouse is moved.
        If self.mouse_pressed is True, it updates the position of the stimuli.
        """
        if self.mouse_pressed:
            # Get the mouse coordinates and pick the cell
            x, y = self.GetInteractor().GetEventPosition()
            self.pick_cell(x, y)

    def pick_cell(self, x, y):
        """
        Function that picks the cell that was clicked and applies a force to it.
        :param x: x coordinate of the mouse
        :param y: y coordinate of the mouse
        """

        # Pick the cell that was clicked
        self.picker.Pick(x, y, 0, self.gui.plotter.renderer)
        cell_id = self.picker.GetCellId()

        # If the cell exists
        if cell_id != -1:
            apply_stimuli_pressure(self.fenics, self.gui, self.stimuli, self.picker, cell_id)
