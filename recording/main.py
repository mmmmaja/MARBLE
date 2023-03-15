import os
from plotter import Plot


# Set to True if data is about to be recorded, otherwise read the data from the path
RECORDING = False

# Path to the folder of the xlsx files
data_folder_path = "C:/University/Marble/Data/correct_orthosis_90/"

# Determine if we want to display linearized or raw data
linearized_data = True

# Name of the file
file_name = '1'

# Set the time of the recording
time_recording = 10

path = os.path.join(data_folder_path, (file_name + "_LIN" if linearized_data else file_name + "_RAW") + ".xlsx")


# Run real-time recording for the arm, show pressure nad angle plots
if RECORDING:
    from arm_recording import run_recording
    run_recording(data_folder_path, file_name, time_recording)

# Plot data from the indicated folder and file
else:
    plot = Plot(time_recording)
    plot.display_data(path)

