import pyvistaqt as pvqt  # For updating plots real time
from AI.fenics import *
from AI.mesh_converter import *
from AI.material_handler import *
from PyQt5.QtWidgets import QApplication


def add_mesh(mesh, plotter, fenics):
    """
    Adds the mesh in the .vtk format to the plotter
    """
    _visual_properties = fenics.rank_material.visual_properties
    plotter.add_mesh(
        mesh,
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


def apply_volume_force(mesh, plotter, fenics):
    """
    Function that applies a volume (stable) force across the whole mesh
    :param fenics: SfePy solver
    :param plotter: PyVista plotter
    :param mesh: Mesh object including the Pyvista mesh
    """

    u = fenics.apply_volume_force(F=1.9)
    mesh.update(u)

    # Reloading the mesh to the scene
    plotter.clear()
    add_mesh(mesh.vtk_mesh, plotter, fenics)

    # Redrawing
    plotter.update()


def apply_cell_force(cell, fenics, plotter, mesh):
    """
    Function that applies a force to the vertex that was picked
    :param cell: cell object that was picked form the callback
    :param fenics: SfePy solver
    :param plotter: PyVista plotter
    :param mesh: Mesh object including the Pyvista mesh
    """
    print("Activated cell: ", cell.points)
    vertex_ids = mesh.get_vertex_ids_from_coords(cell.points)
    print("Activated vertices: ", vertex_ids)

    u = fenics.apply_vertex_specific_force(vertex_ids, F=1.9)
    mesh.update(u)
    # Reloading the mesh to the scene
    plotter.clear()
    add_mesh(mesh.vtk_mesh, plotter, fenics)

    # Redrawing
    plotter.update()


app = QApplication(sys.argv)

mesh_boost = GridMesh(30, 30, z_function=wave)
_fenics = FENICS(mesh_boost, silicon)

"""
Background plotter

From documentation:
PyVista provides a plotter that enables users to create a rendering window in the background,
that remains interactive while the user performs their processing.

See documentation for more details:
https://qtdocs.pyvista.org/api_reference.html#pyvistaqt.BackgroundPlotter
"""

_plotter = pvqt.BackgroundPlotter()

# Add the mesh to the plotter
add_mesh(mesh_boost.vtk_mesh, _plotter, _fenics)

# Add the material name to the plotter
text = _fenics.rank_material.name + \
       '\nE: ' + str(_fenics.rank_material.young_modulus) + \
       '\nv: ' + str(_fenics.rank_material.poisson_ratio)
_plotter.add_text(text, position='lower_left', font_size=8, color='white', shadow=True)

# Add the event on the press of the space bar
_plotter.add_key_event("space", lambda: apply_volume_force(mesh_boost, _plotter, _fenics))

_plotter.enable_cell_picking(
    callback=lambda cell: apply_cell_force(cell, _fenics, _plotter, mesh_boost),
    font_size=10,
    color='white',
    point_size=30,
    style='wireframe',
    line_width=5,
    through=False
)

_plotter.showFullScreen()
app.exec_()
