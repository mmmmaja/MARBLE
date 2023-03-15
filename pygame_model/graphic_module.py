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


class Rectangle:

    def __init__(self, pos, a, b):
        self.pos = pos
        self.a = a
        self.b = b
        self.alpha = 20  # Starting alpha value
        self.color = hex2RGB("#4ee96e")

    def draw(self, surface):
        points = [
            [self.pos[0] - self.a/2, self.pos[1] - self.b / 2],
            [self.pos[0] + self.a/2, self.pos[1] - self.b / 2],
            [self.pos[0] + self.a/2, self.pos[1] + self.b / 2],
            [self.pos[0] - self.a/2, self.pos[1] + self.b / 2],
        ]
        # Draw the circle with the current alpha value
        gfxdraw.filled_polygon(
            surface,
            points,
            (self.color[0], self.color[1], self.color[2], self.alpha)
        )

        # Decrease the alpha value for the next frame
        self.alpha -= 0.2
