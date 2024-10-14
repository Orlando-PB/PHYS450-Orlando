import os
import numpy as np
from astropy.io import fits

def load_combined_fits(base_folder, keyword):
    """
    Load and combine FITS files matching a keyword (e.g., 'Dark', 'Bias', 'Flat').
    Averages the data from all matching files.
    
    :param base_folder: Folder where FITS files are located
    :param keyword: Keyword to match files (e.g., 'Dark', 'Bias', 'Flat')
    :return: Combined (averaged) FITS data as a 2D numpy array
    """
    fits_files = [f for f in os.listdir(base_folder) if keyword in f and f.endswith('.fit')]
    
    if not fits_files:
        raise FileNotFoundError(f"No FITS files found for keyword '{keyword}' in {base_folder}")
    
    combined_data = None

    for file in fits_files:
        file_path = os.path.join(base_folder, file)
        with fits.open(file_path) as hdul:
            data = hdul[0].data.astype(np.float32)  # Convert to float32 for precision
            if combined_data is None:
                combined_data = data
            else:
                combined_data += data

    combined_data /= len(fits_files)  # Average the data
    return combined_data
