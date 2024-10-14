import os
from fits_processor import process_light_images
from gui import run_gui

# Define the base folder where all files are located
base_folder = "./demo"
output_folder = os.path.join(base_folder, "Output")

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if __name__ == "__main__":
    # Launch the GUI
    run_gui(base_folder, output_folder, process_light_images)
