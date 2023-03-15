import math
import sys
import pygame
from pygame import gfxdraw
import mesh

FRAME_WIDTH, FRAME_HEIGHT = 1000, 500


def hex2RGB(color):
    """
    :param color: in the hex format
    :return: [R, G, B] values in range (0, 1)
    """
    color = color.lstrip('#')
    R, G, B = int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)
    rgb = (R, G, B)
    return rgb


class Display:

    def __init__(self, sensor_mesh):

        self.sensor_mesh = sensor_mesh
        self.mouse_pressed = False

        # Stores press locations to display disappearing circles
        self.presses = []

        self.screen = pygame.display.set_mode(
            size=(FRAME_WIDTH, FRAME_HEIGHT)
        )
        self.update()
        pygame.display.update()

    def run(self):
        while True:
            self.detect_events()
            self.draw_circles()

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
        for v in self.sensor_mesh.vertices:
            pygame.draw.circle(self.screen, hex2RGB("#2ca5ff"), v, 4)

        for s in self.sensor_mesh.SENSOR_ARRAY:
            if s.activated:
                pygame.draw.circle(self.screen, hex2RGB("#a956ff"), s.position, 6)

    def detect_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_pressed = True

            if event.type == pygame.MOUSEBUTTONUP:
                self.mouse_pressed = False

            if self.mouse_pressed:
                pos = pygame.mouse.get_pos()
                # Add a new circle to the list when the mouse is clicked
                self.presses.append(Circle(pos))

                # Change pressure outputs of the sensors
                self.sensor_mesh.press(list(pos))

            # TODO
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.sensor_mesh.get_values()

    def draw_circles(self):
        self.update()
        for circle in self.presses:

            circle.draw(self.screen)

            # Remove the circle from the list if it has become invisible
            if circle.alpha <= 0:
                self.presses.remove(circle)
        pygame.display.update()


class Circle:

    def __init__(self, pos):
        self.pos = pos
        self.radius = 30
        self.alpha = 20  # Starting alpha value
        self.color = hex2RGB("#4ee96e")

    def draw(self, surface):
        # Draw the circle with the current alpha value
        gfxdraw.filled_circle(
            surface,
            self.pos[0], self.pos[1], self.radius,
            (self.color[0], self.color[1], self.color[2], self.alpha)
        )

        # Decrease the alpha value for the next frame
        self.alpha -= 0.2


display = Display(
    mesh.Mesh(width=10, height=10, center=(FRAME_WIDTH / 4, FRAME_HEIGHT / 2))
)
display.run()
