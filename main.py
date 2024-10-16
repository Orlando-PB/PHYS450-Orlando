import os
from datetime import datetime
from fits_processor import process_light_images
from gui import run_gui

# Define the base folder where all files are located
base_folder = "./demo"

# Create a unique output folder name using the current date and time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = os.path.join(base_folder, f"Output_{timestamp}")

# Ensure the unique output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if __name__ == "__main__":
    # Launch the GUI
    run_gui(base_folder, output_folder, process_light_images)
