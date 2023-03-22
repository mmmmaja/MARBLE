import pygame
import mesh
import stimulis
from stimulis import hex2RGB, ForgeRecording, Record
from mesh import UNIT
from deformation_function import *

FRAME_WIDTH, FRAME_HEIGHT = 1000, 500


class Display:

    def __init__(self, sensor_mesh, stimuli):

        self.sensor_mesh = sensor_mesh
        self.stimuli = stimuli

        # Indicates if currently the mouse is pressed
        self.mouse_pressed = False

        # Indicates if currently the pressure data is recorded
        self.recording = False

        # Stores press locations to display disappearing circles
        self.presses = []

        # Index of the X line to be displayed on the right side od the visualization
        self.LINE_INDEX = 2

        # Number of centimeters to shift right visualization up
        self.D_Y = 2

        # Press to start or stop recording
        self.record_button, self.forge_recording = None, None

        self.screen = pygame.display.set_mode(
            size=(FRAME_WIDTH, FRAME_HEIGHT)
        )
        pygame.init()

        self.update_central_section()
        self.update_cross_section()
        self.draw_settings()
        pygame.display.update()

    def run(self):

        UPDATE_INTERVAL = 1000  # Update every n milliseconds
        clock = pygame.time.Clock()
        ticks = 0

        while True:
            self.detect_events()
            self.display_presses()

            if (pygame.time.get_ticks() - ticks) > UPDATE_INTERVAL:
                ticks = pygame.time.get_ticks()
                if self.recording:
                    self.sensor_mesh.append_data()

            clock.tick(60)

    def draw_settings(self):
        # Add record button
        self.record_button = Record(self.screen, position=(FRAME_WIDTH // 2 + 30, 50))
        self.record_button.add()

        # TODO add 'forge recording' button
        self.forge_recording = ForgeRecording(self.screen, position=(FRAME_WIDTH // 2 + 150, 50))
        self.forge_recording.add()

    def update_cross_section(self):

        # Draw section background
        rect = pygame.Rect(
            FRAME_WIDTH / 2, FRAME_HEIGHT // 2 - (self.D_Y + 1) * UNIT,
            FRAME_WIDTH / 2, FRAME_HEIGHT)
        pygame.draw.rect(self.screen, hex2RGB("#3f4152"), rect)

        # Draw Y axis
        x = FRAME_WIDTH // 2
        color, line_width = hex2RGB("#0d0e12"), 2
        while x <= FRAME_WIDTH:
            pygame.draw.line(
                self.screen, color,
                (x, FRAME_HEIGHT // 2 - self.D_Y * UNIT), (x, FRAME_HEIGHT), line_width)
            x += UNIT
            color, line_width = hex2RGB("#31343d"), 1
        # Draw X axis
        y = FRAME_HEIGHT // 2
        color, line_width = hex2RGB("#0d0e12"), 2
        while y - UNIT * self.D_Y <= FRAME_HEIGHT:
            pygame.draw.line(
                self.screen, color,
                (FRAME_WIDTH // 2, y - self.D_Y * UNIT), (FRAME_WIDTH, y - self.D_Y * UNIT), line_width)
            y += UNIT
            color, line_width = hex2RGB("#31343d"), 1

        # Draw function of the deformation
        curve_points = []

        # Append the pressure points from the mesh object
        sensor_line = self.sensor_mesh.get_points_along_X(self.LINE_INDEX)
        for i in range(len(sensor_line)):
            x = sensor_line[i].frame_position[0]
            y = - sensor_line[i].deformation
            curve_points.append((x + FRAME_WIDTH // 2, y + FRAME_HEIGHT // 2 - self.D_Y * UNIT))

        pygame.draw.lines(self.screen, hex2RGB("#4ee96e"), False, curve_points, 2)

    def update_central_section(self):
        rect = pygame.Rect(0, 0, FRAME_WIDTH / 2, FRAME_HEIGHT)
        pygame.draw.rect(self.screen, hex2RGB("#262833"), rect)

        for t in self.sensor_mesh.triangles:
            pygame.draw.polygon(self.screen, hex2RGB("#2d2f3d"), t, 2)

        # Mark the line that is displayed on the cross_section
        sensor_line = self.sensor_mesh.get_points_along_X(self.LINE_INDEX)
        pygame.draw.line(
            self.screen,
            hex2RGB("#9c82dd"),
            sensor_line[0].frame_position,
            sensor_line[len(sensor_line) - 1].frame_position,
            2)

        # Display the sensors
        for s in self.sensor_mesh.SENSOR_ARRAY:
            circle_prop = s.get_circle_properties()
            pygame.draw.circle(self.screen, circle_prop[0], circle_prop[1], circle_prop[2])

    def detect_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.sensor_mesh.save_data()
                quit()

            # Record the pressure from the mouse input
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_pressed = True
                if self.record_button.button_rect.collidepoint(event.pos):
                    self.record_button.add()
                    self.recording = not self.recording
                if self.forge_recording.button_rect.collidepoint(event.pos):
                    self.forge_recording.add()

            if event.type == pygame.MOUSEBUTTONUP:
                self.mouse_pressed = False

            if self.mouse_pressed:
                pos = np.array(pygame.mouse.get_pos())
                if pos[0] < FRAME_WIDTH // 2 - UNIT:
                    self.stimuli.set_position(pos)

                    # Add a new circle to the list when the mouse is clicked
                    self.presses.append(
                        self.stimuli.get_shape()
                    )

                    self.stimuli.set_deformation(-2 * UNIT)
                    # Change pressure outputs of the sensors
                    self.sensor_mesh.press(self.stimuli)

            # Change the index of the line shown on the cross-section with the arrows
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.LINE_INDEX = max(self.LINE_INDEX - 1, 0)
                if event.key == pygame.K_DOWN:
                    self.LINE_INDEX = min(self.LINE_INDEX + 1, self.sensor_mesh.height - 1)

    def display_presses(self):
        self.update_central_section()
        self.update_cross_section()
        for shape in self.presses:
            shape.draw(self.screen)

            # Remove the circle from the list if it has become invisible
            if shape.alpha <= 0:
                self.presses.remove(shape)
        pygame.display.update()


rectangle_stimuli = stimulis.Cuboid(DeformationFunction(), 2 * UNIT, 2 * UNIT)
sphere_stimuli = stimulis.Sphere(DeformationFunction(), UNIT)

display = Display(
    mesh.Mesh(width=10, height=10, center=(FRAME_WIDTH / 4, FRAME_HEIGHT / 2)),
    sphere_stimuli
)
display.run()
