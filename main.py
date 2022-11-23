import os
from plotter import Plot

"""
make plot for lukas
get neighbouring sensors
    input, output: indexes
maja GUI
"""

# Set to True if data is about to be recorded, otherwise read the data from the path
RECORDING = False

# Path to the folder of the xlsx files
data_folder_path = 'C:/Users/majag/Desktop/marble/data/correct_orthosis_90'

# Determine if we want to display linearized or raw data
linearized_data = True

# Name of the file
file_name = '1'

# Set the time of the recording
time_recording = 10

path = os.path.join(data_folder_path, (file_name + "_LIN" if linearized_data else file_name + "_RAW") + ".xlsx")


if RECORDING:
    from arm_recording import run_recording
    run_recording(data_folder_path, file_name, time_recording)

else:
    # Plot imported data
    plot = Plot(time_recording)
    plot.display_data(path)

"""
TODO 

fix raw plotting (normalize?)
GUI: 
    choose lin/raw data, 
    button start recording
    add plot lines
main 
    start of the program
    time recording as param


do repository github

Lukas:
data storage
"""


