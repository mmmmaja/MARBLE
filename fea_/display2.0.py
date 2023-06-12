import sys
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from fea_ import fea_main
from fea_main import FEA, Material
from model import advanced_mesh

no_material = (0.0, 0.0, 0.0, 1.0)
ambient_material = (0.7, 0.7, 0.7, 1.0)
ambient_material_color = (0.8, 0.8, 0.2, 1.0)
diffuse_material = (0.1, 0.5, 0.8, 1.0)
specular_material = (1.0, 1.0, 1.0, 1.0)
no_shininess = 0.0
low_shininess = 5.0
high_shininess = 100.0
emission_material = (0.3, 0.2, 0.2, 0.0)


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
        centre = self.fea.mesh.get_central_point()
        return [centre[0], centre[1], 0]

    def load_sensor_material(self):
        glMaterialfv(GL_FRONT, GL_AMBIENT, no_material)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse_material)
        glMaterialfv(GL_FRONT, GL_SPECULAR, specular_material)
        glMaterialf(GL_FRONT, GL_SHININESS, high_shininess)
        glMaterialfv(GL_FRONT, GL_EMISSION, no_material)

    def load_mesh_material(self):
        glMaterialfv(GL_FRONT, GL_AMBIENT, ambient_material_color)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse_material)
        glMaterialfv(GL_FRONT, GL_SPECULAR, specular_material)
        glMaterialf(GL_FRONT, GL_SHININESS, low_shininess)
        glMaterialfv(GL_FRONT, GL_EMISSION, no_material)

    def draw(self):
        self.load_mesh_material()
        # Draw triangles
        for element in self.elements:
            glBegin(GL_TRIANGLES)
            for node in element.nodes:
                glVertex3fv((node.location_3D - self.translation))
            glEnd()

        # Draw points (vertices)
        glPointSize(7.0)  # Set point size to 5 pixels
        self.load_sensor_material()
        glBegin(GL_POINTS)
        for element in self.elements:
            for node in element.nodes:
                glVertex3fv((node.location_3D - self.translation))
        glEnd()

        # Draw the mesh lines
        for element in self.elements:
            glBegin(GL_LINES)
            glVertex3fv((element.nodes[0].location_3D - self.translation))
            glVertex3fv((element.nodes[1].location_3D - self.translation))
            glVertex3fv((element.nodes[2].location_3D - self.translation))
            glVertex3fv((element.nodes[0].location_3D - self.translation))
            glEnd()


class DisplayHandler:

    def __init__(self):

        self.mouse_down = False
        glTranslatef(0.0, 0.0, -20)
        self.load_light()

    def load_light(self):
        # Global light properties
        position = (10.0, 10.0, 0.3, 0.0)
        diffuse = hex_to_rgb('#c1e1ff')
        ambient = (0.5, 0.5, 0.5, 1.0)

        glEnable(GL_LIGHTING)

        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (diffuse[0], diffuse[1], diffuse[2], 0.6))
        glLightfv(GL_LIGHT0, GL_POSITION, position)

        glEnable(GL_LIGHT0)

    def handle_display(self):

        ROTATION_SPEED = 0.1
        SCROLL_SPEED = 0.5

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_down = True
            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouse_down = False

            # Mouse motion on click for rotation

            if event.type == pygame.MOUSEMOTION and self.mouse_down:
                x, y = event.rel
                glRotatef(y * ROTATION_SPEED, 1, 0, 0)
                glRotatef(x * ROTATION_SPEED, 0, 1, 0)

            # Mouse wheel for zooming
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    # Scroll up
                    glTranslatef(0, 0, SCROLL_SPEED)
                elif event.button == 5:
                    # Scroll down
                    glTranslatef(0, 0, - SCROLL_SPEED)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    glTranslatef(-0.5, 0, 0)
                if event.key == pygame.K_RIGHT:
                    glTranslatef(0.5, 0, 0)

                if event.key == pygame.K_UP:
                    glTranslatef(0, 1, 0)
                if event.key == pygame.K_DOWN:
                    glTranslatef(0, -1, 0)

        # Clear the screen and depth buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Reset the view and set the camera position and orientation
        # glLoadIdentity()


def main(fea):
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    handler = DisplayHandler()
    mesh = MeshHandler(fea)

    while True:
        handler.handle_display()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        mesh.draw()
        pygame.display.flip()
        pygame.time.wait(10)


if __name__ == "__main__":
    silicon = Material(density=2.329, young_modulus=140.0, poisson_ratio=0.265)

    # Apply Boundary Conditions
    # Ku = F where u is the unknown displacement vector of all nodes

    _mesh = advanced_mesh.RectangleMesh(10, 10)
    # _mesh = advanced_mesh.csvMesh('C:/Users/majag/Desktop/marble/MARBLE/model/meshes_csv/web.csv')

    main(FEA(_mesh, silicon))
