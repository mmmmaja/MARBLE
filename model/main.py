import advanced_mesh
import stimulis
from stimulis import *
from deformation_function import *
import simulation

FRAME_WIDTH, FRAME_HEIGHT = 1000, 500
UPDATE_INTERVAL = 100  # Update every n milliseconds


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

        # Button objects
        self.record_button, self.forge_recording_button, self.display_recording_button = None, None, None
        self.evaluate_recording = None

        self.displayed_recording = None

        self.screen = pygame.display.set_mode(
            size=(FRAME_WIDTH, FRAME_HEIGHT)
        )
        # Make the window resizable false
        pygame.init()

        self.update_central_section()
        self.update_cross_section()
        self.draw_settings()
        pygame.display.update()

    def run(self):

        clock = pygame.time.Clock()
        ticks = 0

        while True:
            # Detect mouse and key presses
            self.detect_events()
            self.display_presses()

            # If data is currently recorded, then save current mesh values
            if (pygame.time.get_ticks() - ticks) > UPDATE_INTERVAL:
                ticks = pygame.time.get_ticks()
                if self.recording:
                    self.sensor_mesh.append_data()

                if self.displayed_recording:
                    if not self.displayed_recording.read(self.sensor_mesh):
                        self.displayed_recording = None
                        self.display_recording_button.add()

            clock.tick(60)

    def draw_settings(self):
        # Fill the background
        rect = pygame.Rect(
            FRAME_WIDTH // 2, 0,
            FRAME_WIDTH, FRAME_HEIGHT // 2)
        pygame.draw.rect(self.screen, hex2RGB("#181a21"), rect)

        # Add 'Record' button
        self.record_button = RecordButton(self.screen, position=(FRAME_WIDTH // 2 + 30, 20))
        self.record_button.add()

        # Add 'forge recording' button
        self.forge_recording_button = ForgeRecordingButton(
            self.screen, position=(FRAME_WIDTH // 2 + 150, 20), width=130
        )
        self.forge_recording_button.add()

        # Add 'Read recording' button
        self.display_recording_button = DisplayRecordingButton(
            self.screen, position=(FRAME_WIDTH // 2 + 300, 20), width=130
        )
        self.display_recording_button.add()

        # Add 'Evaluate recording' button
        self.evaluate_recording = EvaluateRecordingButton(
            self.screen, position=(FRAME_WIDTH // 2 + 30, 80), width=130
        )
        self.evaluate_recording.add()

    def update_cross_section(self):

        # Number of centimeters to shift right visualization up
        D_Y = 2

        # Draw section background
        rect = pygame.Rect(
            FRAME_WIDTH / 2, FRAME_HEIGHT // 2 - (D_Y + 1) * UNIT,
            FRAME_WIDTH / 2, FRAME_HEIGHT)
        pygame.draw.rect(self.screen, hex2RGB("#3f4152"), rect)

        # Draw Y axis
        x = FRAME_WIDTH // 2
        color, line_width = hex2RGB("#0d0e12"), 2
        while x <= FRAME_WIDTH:
            pygame.draw.line(
                self.screen, color,
                (x, FRAME_HEIGHT // 2 - D_Y * UNIT), (x, FRAME_HEIGHT), line_width)
            x += UNIT
            color, line_width = hex2RGB("#31343d"), 1

        # Draw X axis
        y = FRAME_HEIGHT // 2
        color, line_width = hex2RGB("#0d0e12"), 2
        while y - UNIT * D_Y <= FRAME_HEIGHT:
            pygame.draw.line(
                self.screen, color,
                (FRAME_WIDTH // 2, y - D_Y * UNIT), (FRAME_WIDTH, y - D_Y * UNIT), line_width)
            y += UNIT
            color, line_width = hex2RGB("#31343d"), 1

        # Draw function of the deformation
        curve_points = []
        # Append the pressure points from the mesh object
        sensor_line = self.sensor_mesh.displayed_points
        for i in range(len(sensor_line)):
            x = sensor_line[i].frame_position[0]
            y = - sensor_line[i].deformation * UNIT
            curve_points.append((x + FRAME_WIDTH // 2, y + FRAME_HEIGHT // 2 - D_Y * UNIT))
        pygame.draw.lines(self.screen, hex2RGB("#4ee96e"), False, curve_points, 2)

    def update_central_section(self):
        # Fill the background
        rect = pygame.Rect(0, 0, FRAME_WIDTH / 2, FRAME_HEIGHT)
        pygame.draw.rect(self.screen, hex2RGB("#262833"), rect)

        # Display the mesh of the sensor array
        for t in self.sensor_mesh.triangles:
            pygame.draw.polygon(self.screen, hex2RGB("#2d2f3d"), t, 2)

        # Mark the line that is displayed on the cross_section
        sensor_line = self.sensor_mesh.displayed_points
        pygame.draw.line(
            self.screen,
            hex2RGB("#9c82dd"),
            sensor_line[0].frame_position,
            sensor_line[len(sensor_line) - 1].frame_position,
            2)

        # Display the sensors as small circles
        for s in self.sensor_mesh.SENSOR_ARRAY:
            # Colors corresponds to pressure values
            circle_prop = s.get_circle_properties()
            pygame.draw.circle(self.screen, circle_prop[0], circle_prop[1], circle_prop[2])

    def detect_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

            # Display current user actions
            if not self.displayed_recording:

                # Record the pressure from the mouse input
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.mouse_pressed = True

                    # 'Record' button was clicked
                    if self.record_button.button_rect.collidepoint(event.pos):
                        self.record_button.add()
                        self.recording = not self.recording
                        if self.record_button.STATE == 1:
                            print("Your data was saved")
                            self.sensor_mesh.save_data()

                    # 'Forge recording' button was clicked
                    if self.forge_recording_button.button_rect.collidepoint(event.pos):
                        simulation.ForgeRecording(
                            frame_dim=[FRAME_WIDTH // 2, FRAME_HEIGHT],
                            stimuli=self.stimuli, sensor_mesh=self.sensor_mesh,
                            update_interval=UPDATE_INTERVAL
                        )
                        self.forge_recording_button.add()

                    # 'Read recording' button was clicked
                    if self.display_recording_button.button_rect.collidepoint(event.pos):
                        self.display_recording_button.add()
                        self.displayed_recording = simulation.ReadRecording()

                    # 'Read recording' button was clicked
                    if self.evaluate_recording.button_rect.collidepoint(event.pos):
                        self.evaluate_recording.add()
                        simulation.evaluate_recording()

                if event.type == pygame.MOUSEBUTTONUP:
                    self.mouse_pressed = False

                # Sensor mesh is pressed
                if self.mouse_pressed:
                    pos = np.array(pygame.mouse.get_pos())
                    if pos[0] < FRAME_WIDTH // 2 - UNIT:
                        stimuli_position = (np.concatenate([pos, np.array([0])]) - OFFSET) / UNIT
                        self.stimuli.set_position(stimuli_position)

                        # Add a new circle to the list when the mouse is clicked
                        self.presses.append(self.stimuli.get_shape())
                        self.stimuli.set_deformation(-2)
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


rectangle_stimuli = stimulis.Cuboid(DeformationFunction(), 2, 2)
sphere_stimuli = stimulis.Sphere(DeformationFunction(), 1)

# rect_mesh = advanced_mesh.RectangleMesh(10, 10)
csv_mesh = advanced_mesh.csvMesh('meshes_csv/web.csv')

display = Display(csv_mesh, sphere_stimuli)
display.run()


