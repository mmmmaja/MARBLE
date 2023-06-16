import mouseinfo
import numpy as np
import ursina
from ursina import *
from ursina import Mesh
from fea_main import FEA, Material
from advanced_mesh_2 import *
from stimuli_2 import *
from projection import *

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600


class MeshHandler:

    def __init__(self, fea):
        self.fea = fea
        self.elements = self.fea.elements
        self.translation = self.get_translation()

    def get_translation(self):
        # So that the mesh will be in the center of the frame and rotation runs smoothly
        centre = self.fea.mesh.get_central_point()
        return [centre[0], centre[1], 0]

    def draw(self):
        for element in self.elements:
            vertices = []
            for node in element.nodes:
                vertices.append(tuple(node.position))

            model = Entity(model=ursina.Mesh(vertices=vertices, mode='triangle'))
            lines = Entity(model=ursina.Mesh(vertices=vertices, mode='line', thickness=1), color=color.cyan, z=-1)
            points = Entity(model=ursina.Mesh(vertices=vertices, mode='point', thickness=.05), color=color.magenta,
                            z=-1.01)

            # Add the material to the mesh


class StimuliHandler:

    def __init__(self):
        self.stimuli = []
        self.translation = self.get_translation()

    def draw(self):
        pass


# Updates every frame
def update():
    ROTATION_SPEED = 5.5
    TRANSLATION_SPEED = 5
    # Translation of the camera events

    # Move the camera left if the left arrow key is pressed
    if held_keys['left arrow']:
        camera.x -= TRANSLATION_SPEED * time.dt
    # Move the camera right if the right arrow key is pressed
    if held_keys['right arrow']:
        camera.x += TRANSLATION_SPEED * time.dt
    # Move the camera up if the up arrow key is pressed
    if held_keys['up arrow']:
        camera.y += TRANSLATION_SPEED * time.dt
    # Move the camera down if the down arrow key is pressed
    if held_keys['down arrow']:
        camera.y -= TRANSLATION_SPEED * time.dt

    # Add scrolling functionality
    if held_keys['q']:
        camera.z += TRANSLATION_SPEED * time.dt
    if held_keys['e']:
        camera.z -= TRANSLATION_SPEED * time.dt

    # Rotation of the camera events
    # Rotate the camera around the z-axis
    if held_keys['a']:
        camera.rotation_z += ROTATION_SPEED * time.dt
    if held_keys['d']:
        camera.rotation_z -= ROTATION_SPEED * time.dt
    # Rotate the camera around the x-axis
    if held_keys['w']:
        camera.rotation_x += ROTATION_SPEED * time.dt


# cube = Entity(model='cube', color=color.orange, scale=(1, 2, 3), position=(1, 2, 3))
#
#
# sphere = Entity(model='sphere', color=color.white, scale=(1, 2, 3), position=(1, 2, 3))


silicon = Material(density=2.329, young_modulus=140.0, poisson_ratio=0.265, thickness=1.25)
_mesh = RectangleMesh(10, 10, z_function=concave)
_stimuli = Sphere(radius=5.5)
fea = FEA(_mesh, silicon, _stimuli)

# Create a mesh


app = Ursina()
camera.position = (0, 0, -20)
mesh_handler = MeshHandler(fea)
mesh_handler.draw()

app.run()
