import os
from plotter import Plot

# Path of the xlsx files folder
data_folder_path = 'incorrect_orthosis_up_1cm_45'
# Determine if we want to display linearized or raw data
linearized_data = True
# Name of the file
file_name = "1"


plot = Plot()
path = os.path.join('data', data_folder_path, (file_name + "_LIN" if linearized_data else file_name + "_RAW") + ".xlsx")
plot.display_data(path)
