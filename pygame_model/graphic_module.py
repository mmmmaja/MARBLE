import pygame
from pygame import gfxdraw


def hex2RGB(color):
    """
    :param color: in the hex format
    :return: [R, G, B] values in range (0, 1)
    """
    color = color.lstrip('#')
    R, G, B = int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)
    rgb = (R, G, B)
    return rgb


class Circle:

    def __init__(self, pos, radius):
        self.pos = pos
        self.radius = radius
        self.alpha = 15  # Starting alpha value
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


class Rectangle:

    def __init__(self, pos, a, b):
        self.pos = pos
        self.a = a
        self.b = b
        self.alpha = 15  # Starting alpha value
        self.color = hex2RGB("#4ee96e")

    def draw(self, surface):
        points = [
            [self.pos[0] - self.a / 2, self.pos[1] - self.b / 2],
            [self.pos[0] + self.a / 2, self.pos[1] - self.b / 2],
            [self.pos[0] + self.a / 2, self.pos[1] + self.b / 2],
            [self.pos[0] - self.a / 2, self.pos[1] + self.b / 2],
        ]
        # Draw the circle with the current alpha value
        gfxdraw.filled_polygon(
            surface,
            points,
            (self.color[0], self.color[1], self.color[2], self.alpha)
        )

        # Decrease the alpha value for the next frame
        self.alpha -= 0.2


class Button:

    def __init__(self, screen, position, width=100, height=40):
        self.button_surface = None
        self.button_rect = None

        self.screen = screen
        self.position = position
        self.font = pygame.font.SysFont(None, 20)
        self.width, self.height = width, height

        self.STATE = 0

        self.create()

    def create(self):
        self.button_rect = pygame.Rect(
            self.position[0], self.position[1],
            self.width, self.height
        )
        self.button_surface = pygame.Surface((self.width, self.height))


class RecordButton(Button):

    def add(self):

        if self.STATE == 0:

            self.button_surface.fill(hex2RGB('00a2ff'))
            button_text = self.font.render("Record", True, hex2RGB('262834'))
            button_text_rect = button_text.get_rect(center=(self.width // 2, self.height // 2))
            self.button_surface.blit(button_text, button_text_rect)

            # Draw the button
            self.screen.blit(self.button_surface, self.button_rect)

        else:
            self.button_surface.fill(hex2RGB('48b764'))
            button_text = self.font.render("Stop recording", True, hex2RGB('262834'))
            button_text_rect = button_text.get_rect(center=(self.width // 2, self.height // 2))
            self.button_surface.blit(button_text, button_text_rect)
            # Draw the button
            self.screen.blit(self.button_surface, self.button_rect)

        self.STATE = (self.STATE + 1) % 2


class ForgeRecordingButton(Button):

    def add(self):
        if self.STATE == 0:
            self.button_surface.fill(hex2RGB('b45f9e'))
            button_text = self.font.render("Forge a recording", True, hex2RGB('262834'))
            button_text_rect = button_text.get_rect(center=(self.width // 2, self.height // 2))
            self.button_surface.blit(button_text, button_text_rect)

            # Draw the button
            self.screen.blit(self.button_surface, self.button_rect)

        # if self.STATE == 1:
        #     self.button_surface.fill(hex2RGB('005e88'))
        #     self.screen.blit(self.button_surface, self.button_rect)
        #
        # self.STATE = (self.STATE + 1) % 2


class DisplayRecordingButton(Button):

    def add(self):
        if self.STATE == 0:
            self.button_surface.fill(hex2RGB('15906d'))
            button_text = self.font.render("Read recording", True, hex2RGB('262834'))
            button_text_rect = button_text.get_rect(center=(self.width // 2, self.height // 2))
            self.button_surface.blit(button_text, button_text_rect)

            # Draw the button
            self.screen.blit(self.button_surface, self.button_rect)

        if self.STATE == 1:
            self.button_surface.fill(hex2RGB('48b764'))
            self.screen.blit(self.button_surface, self.button_rect)

        self.STATE = (self.STATE + 1) % 2
