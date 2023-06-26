import pyvistaqt as pvqt  # For updating plots real time
from PyQt5.QtWidgets import QApplication
from AI.fenics import *
from AI.mesh_converter import *
from AI.material_handler import *
from PyQt5.QtCore import QObject, QTimer


FORCE = 0.5


def add_mesh(plotter, vtk_mesh, rank_material):
    """
    :param plotter: pyvistaqt plotter object
    :param vtk_mesh: mesh in .vtk format (instance of mesh_boost)
    :param rank_material: Holds all the physical properties of the material of the mesh

    Took it out of the Main to create a separate timer class
    Adds the mesh in the .vtk format to the plotter
    """

    _visual_properties = rank_material.visual_properties
    plotter.add_mesh(
        vtk_mesh,
        show_edges=True,
        smooth_shading=True,
        show_scalar_bar=False,
        edge_color=_visual_properties['edge_color'],
        color=_visual_properties['color'],
        specular=_visual_properties['specular'],
        metallic=_visual_properties['metallic'],
        roughness=_visual_properties['roughness']
    )
    plotter.enable_lightkit()


class Main:

    def __init__(self, mesh_boost, rank_material):
        self.plotter = None
        self.mesh_boost = mesh_boost
        self.rank_material = rank_material
        self.fenics = FENICS(self.mesh_boost, self.rank_material)
        self.create_plot()

        """
        Timer for updating the state during fenics calculations (Starts a new thread)
        Documentation:
        https://doc.qt.io/qtforpython-5/PySide2/QtCore/QTimer.html
        """
        self.timer = None

    def create_plot(self):
        # Creates a plotter object and sets all the initial settings
        self.plotter = pvqt.BackgroundPlotter()
        add_mesh(self.plotter, self.mesh_boost.vtk_mesh, self.rank_material)

        # Add the material name to the plotter
        text = self.rank_material.name + \
            '\nE: ' + str(self.rank_material.young_modulus) + \
            '\nv: ' + str(self.rank_material.poisson_ratio)
        self.plotter.add_text(text, position='lower_left', font_size=8, color='white', shadow=True)

        # Enable cell picking
        self.plotter.enable_cell_picking(
            callback=self.apply_force,
            font_size=10,
            color='white',
            point_size=30,
            style='wireframe',
            line_width=5,
            through=False
        )

        # Add the event on the press of the space bar
        self.plotter.add_key_event('space', self.apply_force)

        self.plotter.show()

    def apply_force(self, cell=None, F=FORCE):
        """
        Function that applies a vertex specific force or volume (stable) force across the whole mesh
        """
        if cell is not None:
            vertex_ids = self.mesh_boost.get_vertex_ids_from_coords(cell.points)
            print("Triggered cell force")

        else:
            vertex_ids = None
            print("Triggered volume force")

        # Immediately apply the force to the body (might change it later and push it to the loop)
        print("FORCE applied: ", F)
        # Calculate the displacement
        u = self.fenics.apply_force(vertex_ids, F)
        # update the mesh
        self.mesh_boost.update(u)
        self.plotter.clear()
        add_mesh(self.plotter, self.mesh_boost.vtk_mesh, self.rank_material)
        self.plotter.update()


app = QApplication(sys.argv)
_mesh_boost = GridMesh(30, 30, z_function=wave)
Main(_mesh_boost, rubber)
app.exec_()
