import pygame
from pygame.locals import *
from fea_main import FEA, Material
from advanced_mesh_2 import *
from stimuli_2 import *
from projection import *

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600

# Just bunch of settings for material, clean later
no_material = (0.0, 0.0, 0.0, 1.0)
ambient_material = (0.7, 0.7, 0.7, 1.0)
ambient_material_color = (0.8, 0.8, 0.2, 1.0)
diffuse_material = (0.1, 0.5, 0.8, 1.0)
specular_material = (1.0, 1.0, 1.0, 1.0)
no_shininess = 0.0
low_shininess = 5.0
high_shininess = 100.0
emission_material = (0.3, 0.2, 0.2, 0.0)


def load_mesh_material():
    glMaterialfv(GL_FRONT, GL_AMBIENT, ambient_material_color)
    glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse_material)
    glMaterialfv(GL_FRONT, GL_SPECULAR, specular_material)
    glMaterialf(GL_FRONT, GL_SHININESS, low_shininess)
    glMaterialfv(GL_FRONT, GL_EMISSION, no_material)


def load_sensor_material():
    glMaterialfv(GL_FRONT, GL_AMBIENT, no_material)
    glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse_material)
    glMaterialfv(GL_FRONT, GL_SPECULAR, specular_material)
    glMaterialf(GL_FRONT, GL_SHININESS, high_shininess)
    glMaterialfv(GL_FRONT, GL_EMISSION, no_material)


def load_light():
    # Global light properties
    position = (10.0, 10.0, 0.3, 0.0)
    diffuse = hex_to_rgb('#c1e1ff')
    ambient = (0.5, 0.5, 0.5, 1.0)

    glEnable(GL_LIGHTING)

    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (diffuse[0], diffuse[1], diffuse[2], 0.6))
    glLightfv(GL_LIGHT0, GL_POSITION, position)

    glEnable(GL_LIGHT0)


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    # Color value between 0 and 1
    return [int(value[i:i + lv // 3], 16) / 255 for i in range(0, lv, lv // 3)]


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
        load_mesh_material()
        # Draw triangles
        for element in self.elements:
            glBegin(GL_TRIANGLES)
            for node in element.nodes:
                glVertex3fv((node.sensor.position - self.translation))
            glEnd()

        # Draw points (vertices)
        glPointSize(7.0)  # Set point size to 5 pixels
        load_sensor_material()
        glBegin(GL_POINTS)
        for element in self.elements:
            for node in element.nodes:
                glVertex3fv((node.sensor.position - self.translation))
        glEnd()

        # Draw the mesh lines
        for element in self.elements:
            glBegin(GL_LINES)
            for i in range(4):
                glVertex3fv((element.nodes[i % 3].sensor.position - self.translation))
            glEnd()


def translate(translation_vector):
    glTranslatef(translation_vector[0], translation_vector[1], translation_vector[2])


def rotate(rotation_vector):
    # The glRotatef function multiplies the current matrix by a rotation matrix
    glRotatef(rotation_vector[0], 1, 0, 0)
    glRotatef(rotation_vector[1], 0, 1, 0)
    glRotatef(rotation_vector[2], 0, 0, 1)


class DisplayHandler:

    def __init__(self, fea):

        self.fea = fea

        # Then all the interaction with the visualization happens
        self.mouse_down = False

        # On space bar click change the mode to activation:
        # instead of rotation we can activate mesh with the stimuli
        self.activation = False

        # Set the camera further from the scene
        translate([0.0, 0.0, -20.0])

        # Add some nice lightning to the scene
        load_light()

    def handle_display(self):

        # Constants for interactive events
        ROTATION_SPEED = 0.1
        SCROLL_SPEED = 0.5

        # Mouse motion, activation is enabled
        # Activate the sensor mesh with stimuli
        if self.activation and self.mouse_down:
            # Get mouse position
            pos = pygame.mouse.get_pos()
            self.activate_stimuli(pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            # UPDATE ALL BOOLEANS
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_down = True
            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouse_down = False

            # Rotation and zooming, entire scene and mesh
            # Mouse motion on click for rotation, activation is not enabled
            # Update camera direction
            elif event.type == pygame.MOUSEMOTION and self.mouse_down and not self.activation:
                x, y = event.rel
                # (angle, x, y, z) - angle in DEGREES, x, y, z - coordinates of the vector
                rotation_angle = (y * ROTATION_SPEED, x * ROTATION_SPEED, 0.0)
                rotate(rotation_angle)

            # Mouse wheel for zooming
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    # Scroll up
                    translate([0, 0, SCROLL_SPEED])
                elif event.button == 5:
                    # Scroll down
                    translate([0, 0, -SCROLL_SPEED])

            # Moving scene with arrow keys up, down, left, right
            # Update camera position
            # Apply translation
            translation_vector = [0.0, 0.0, 0.0]
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    translation_vector[0] = -0.5
                if event.key == pygame.K_RIGHT:
                    translation_vector[0] = 0.5
                if event.key == pygame.K_UP:
                    translation_vector[1] = 1.0
                if event.key == pygame.K_DOWN:
                    translation_vector[1] = -1.0

                if event.key == pygame.K_SPACE:
                    self.activation = not self.activation

                translate(translation_vector)

    def activate_stimuli(self, pos):
        scene_pos = get_3D_coordinates(pos)
        # print(pos, '->', scene_pos)
        self.fea.stimuli.set_position(scene_pos)
        # self.fea.apply_pressure()
        self.fea.show_pressure()


def main(fea):
    pygame.init()

    display = (SCREEN_WIDTH, SCREEN_HEIGHT)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    handler = DisplayHandler(fea)
    mesh = MeshHandler(fea)

    while True:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        handler.handle_display()
        mesh.draw()

        pygame.display.flip()
        pygame.time.wait(10)


if __name__ == "__main__":
    silicon = Material(density=2.329, young_modulus=140.0, poisson_ratio=0.265, thickness=1.25)

    # This is grid-like flat mesh
    _mesh = RectangleMesh(5, 5, z_function=concave)
    _mesh = RectangleMesh(10, 10, z_function=concave)
    _mesh = RectangleMesh(12, 12, z_function=convex, sensor_distance=1.5)
    _mesh = CircleMesh(10, 5, z_function=convex, distance_between_nets=2)

    _mesh = RectangleMesh(10, 10, z_function=flat)
    # _mesh = ARM_mesh('model.obj')
    _stimuli = Sphere(radius=5.5)

    # _mesh = csvMesh('C:/Users/majag/Desktop/marble/MARBLE/model/meshes_csv/web.csv')

    main(FEA(_mesh, silicon, _stimuli))
