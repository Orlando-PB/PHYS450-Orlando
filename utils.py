import os
import shutil
import fnmatch
from astropy.io import fits
import numpy as np

def sort_files_into_subfolders(base_folder, output_folder):
    """
    Sort files into subfolders: Lights (separated by filters), Darks, Bias, Flats (separated by filters).
    It searches all subfolders, moves required files, and deletes empty folders afterward.
    Skips folders matching the output folder pattern './base_folder/Output*'.
    """
    categories = {
        "Light": {},
        "Dark": [],
        "Bias": [],
        "Flat": {}
    }

    output_folder_pattern = os.path.join(base_folder, "Output*")

    # Create subfolders for Lights, Darks, Bias, and Flats
    for folder in ['Lights', 'Darks', 'Bias', 'Flats']:
        folder_path = os.path.join(base_folder, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # Function to clean up empty folders
    def delete_empty_folders(folder):
        """
        Delete the folder if it's empty and doesn't match the output pattern.
        """
        if not fnmatch.fnmatch(folder, output_folder_pattern) and not os.listdir(folder):
            os.rmdir(folder)
            print(f"Deleted empty folder: {folder}")

    # Recursively walk through the base folder and subfolders
    for root, _, files in os.walk(base_folder):
        # Skip folders that match the output folder pattern
        if fnmatch.fnmatch(root, output_folder_pattern):
            continue

        for file in files:
            if file.endswith('.fit'):
                file_path = os.path.join(root, file)
                
                # Open FITS file to read the header
                with fits.open(file_path) as hdul: 
                    header = hdul[0].header 
                    image_type = header.get('IMAGETYP', '').strip()  # Get image type (Light, Dark, Bias, Flat)
                    filter_name = header.get('FILTER', 'Unknown').strip()  # Get filter name
                    exposure_time = header.get('EXPTIME', None)  # Get exposure time if needed

                if "Light" in image_type:
                    if filter_name not in categories["Light"]:
                        categories["Light"][filter_name] = []
                        light_folder = os.path.join(base_folder, 'Lights', filter_name)
                        os.makedirs(light_folder, exist_ok=True)
                    categories["Light"][filter_name].append(file)
                    shutil.move(file_path, os.path.join(base_folder, 'Lights', filter_name, file))

                elif "Dark" in image_type:
                    categories["Dark"].append(file)
                    shutil.move(file_path, os.path.join(base_folder, 'Darks', file))

                elif "Bias" in image_type:
                    categories["Bias"].append(file)
                    shutil.move(file_path, os.path.join(base_folder, 'Bias', file))

                elif "Flat" in image_type:
                    if filter_name not in categories["Flat"]:
                        categories["Flat"][filter_name] = []
                        flat_folder = os.path.join(base_folder, 'Flats', filter_name)
                        os.makedirs(flat_folder, exist_ok=True)
                    categories["Flat"][filter_name].append(file)
                    shutil.move(file_path, os.path.join(base_folder, 'Flats', filter_name, file))

        # After processing all files in the folder, delete the folder if it's empty
        delete_empty_folders(root)

    return categories


def calculate_median_frame(frames):
    """
    Calculate the median of a list of FITS frames.

    Parameters:
    frames : list of numpy arrays
        List of FITS frame data to compute the median.

    Returns:
    numpy array : Median frame.
    """
    stacked_frames = np.stack(frames, axis=0)
    median_frame = np.median(stacked_frames, axis=0)
    
    return median_frame

def load_combined_fits(base_folder, category):
    """
    Load the combined master FITS file for the given category (Dark, Bias, Flat).
    Assumes the master FITS file is saved in the relevant folder.
    """
    folder = os.path.join(base_folder, category + 's')
    master_file = os.path.join(folder, f"master_{category.lower()}.fit")

    if not os.path.exists(master_file):
        raise FileNotFoundError(f"Master {category.lower()} file not found.")
    
    with fits.open(master_file) as hdul:
        data = hdul[0].data
    return data
