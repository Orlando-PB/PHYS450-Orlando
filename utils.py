import os
import shutil

def sort_files_into_subfolders(base_folder,output_folder):
    """
    Sort files into subfolders: Lights (separated by filters), Darks, Bias, Flats (separated by filters).
    It searches all subfolders, moves required files, and deletes empty folders afterward.
    """
    categories = {
        "Light": {},
        "Dark": [],
        "Bias": [],
        "Flat": {}
    }

    # Create subfolders for Lights, Darks, Bias, and Flats
    for folder in ['Lights', 'Darks', 'Bias', 'Flats']:
        folder_path = os.path.join(base_folder, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # Function to clean up empty folders
    def delete_empty_folders(folder, output_folder):
        """
        Delete the folder if it's empty, unless it's the skip_folder.
        """
        # Check if the folder is not the one you want to skip
        if os.path.abspath(folder) != os.path.abspath(output_folder) and not os.listdir(folder):
            os.rmdir(folder)
            print(f"Deleted empty folder: {folder}")

                

    # Recursively walk through the base folder and subfolders
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.fit'):
                file_path = os.path.join(root, file)

                if "Light" in file:
                    # Extract filter name
                    filter_name = file.split("_")[4]  # Assuming the filter is the 5th component in the filename
                    if filter_name not in categories["Light"]:
                        categories["Light"][filter_name] = []
                        light_folder = os.path.join(base_folder, 'Lights', filter_name)
                        os.makedirs(light_folder, exist_ok=True)
                    categories["Light"][filter_name].append(file)
                    shutil.move(file_path, os.path.join(base_folder, 'Lights', filter_name, file))

                elif "Dark" in file:
                    categories["Dark"].append(file)
                    shutil.move(file_path, os.path.join(base_folder, 'Darks', file))

                elif "Bias" in file:
                    categories["Bias"].append(file)
                    shutil.move(file_path, os.path.join(base_folder, 'Bias', file))

                elif "Flat" in file:
                    # Extract filter name
                    filter_name = file.split("_")[3]  # Assuming the filter is the 5th component in the filename
                    if filter_name not in categories["Flat"]:
                        categories["Flat"][filter_name] = []
                        flat_folder = os.path.join(base_folder, 'Flats', filter_name)
                        os.makedirs(flat_folder, exist_ok=True)
                    categories["Flat"][filter_name].append(file)
                    shutil.move(file_path, os.path.join(base_folder, 'Flats', filter_name, file))

        # After processing all files in the folder, delete the folder if it's empty
        delete_empty_folders(root,output_folder)

    return categories


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
