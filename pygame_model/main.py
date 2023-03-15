import math
import pygame
import mesh
import stimulis
from stimulis import hex2RGB
from mesh import UNIT
from deformation_function import *

FRAME_WIDTH, FRAME_HEIGHT = 1000, 500


class Display:

    def __init__(self, sensor_mesh, stimuli):

        self.sensor_mesh = sensor_mesh
        self.stimuli = stimuli

        # Indicates if currently the mouse is pressed
        self.mouse_pressed = False

        # Stores press locations to display disappearing circles
        self.presses = []

        self.screen = pygame.display.set_mode(
            size=(FRAME_WIDTH, FRAME_HEIGHT)
        )
        self.update()
        pygame.display.update()

    def run(self):
        # FIXME Add clock here, save the sensor values in the xls file
        while True:
            self.detect_events()
            self.display_presses()

    def update(self):
        self.update_central_section()
        self.update_cross_section()

    def update_cross_section(self):

        rect = pygame.Rect(FRAME_WIDTH / 2, 0, FRAME_WIDTH / 2, FRAME_HEIGHT)
        pygame.draw.rect(self.screen, hex2RGB("#3f4152"), rect)

        # Draw the X and Y axis
        pygame.draw.line(
            self.screen,
            hex2RGB("#0d0e12"),
            (FRAME_WIDTH // 2, FRAME_HEIGHT // 2),
            (FRAME_WIDTH, FRAME_HEIGHT // 2), 2)
        pygame.draw.line(
            self.screen,
            hex2RGB("#0d0e12"),
            (FRAME_WIDTH // 2, 0),
            (FRAME_WIDTH // 2, FRAME_HEIGHT), 2)

        # TODO Draw unit ticks

        self.draw_function()

    def draw_function(self):
        # Generate a series of points along a sine curve
        num_points = 100
        curve_points = []
        for i in range(num_points):
            x = i * (FRAME_WIDTH // 2) / num_points + (FRAME_WIDTH // 2)
            y = FRAME_WIDTH / 2 - 50 * math.sin(i * math.pi / num_points)
            curve_points.append((x, y))

        # Draw the curve
        pygame.draw.lines(self.screen, hex2RGB("#4ee96e"), False, curve_points, 2)

    def update_central_section(self):
        rect = pygame.Rect(0, 0, FRAME_WIDTH / 2, FRAME_HEIGHT)
        pygame.draw.rect(self.screen, hex2RGB("#262833"), rect)

        for t in self.sensor_mesh.triangles:
            pygame.draw.polygon(self.screen, hex2RGB("#2d2f3d"), t, 2)

        for s in self.sensor_mesh.SENSOR_ARRAY:
            circle_prop = s.get_circle_properties()
            pygame.draw.circle(self.screen, circle_prop[0], circle_prop[1], circle_prop[2])

    def detect_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_pressed = True

            if event.type == pygame.MOUSEBUTTONUP:
                self.mouse_pressed = False

            if self.mouse_pressed:
                pos = np.array(pygame.mouse.get_pos())
                self.stimuli.set_position(pos)

                # Add a new circle to the list when the mouse is clicked
                self.presses.append(
                    self.stimuli.get_shape()
                )

                self.stimuli.set_deformation(-2 * UNIT)
                # Change pressure outputs of the sensors
                self.sensor_mesh.press(self.stimuli)

            # TODO
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.sensor_mesh.get_values()

    def display_presses(self):
        self.update()
        for shape in self.presses:
            shape.draw(self.screen)

            # Remove the circle from the list if it has become invisible
            if shape.alpha <= 0:
                self.presses.remove(shape)
        pygame.display.update()


display = Display(
    mesh.Mesh(width=10, height=10, center=(FRAME_WIDTH / 4, FRAME_HEIGHT / 2)),
    # stimuli=stimulis.Cuboid(DeformationFunction(),2 * UNIT, 2 * UNIT)
    stimuli=stimulis.Sphere(DeformationFunction(), UNIT)
)
display.run()
