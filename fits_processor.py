import os
import numpy as np
from astropy.io import fits
from alignment import align_light_frames, save_aligned_frames
from utils import sort_files_into_subfolders, load_combined_fits  # Importing the sorting function from the separate file

def process_light_images(base_folder, output_folder, use_flats=True, use_darks=True, use_biases=True, combine_lrgb=False):
    # First, sort files into subfolders
    sorted_categories = sort_files_into_subfolders(base_folder, output_folder)

    # Create master dark, bias, and flat frames (if needed)
    calibration_folder = os.path.join(output_folder, 'calibration')
    os.makedirs(calibration_folder, exist_ok=True)

    master_dark = master_bias = master_flat = None

    if use_biases:
        bias_folder = os.path.join(base_folder, 'Bias')
        create_master_bias(bias_folder, calibration_folder)
        master_bias = load_master_frame(calibration_folder, "master_bias")

    if use_darks:
        dark_folder = os.path.join(base_folder, 'Darks')
        create_master_dark(dark_folder, calibration_folder, master_bias)
        master_dark = load_master_frame(calibration_folder, "master_dark")

    if use_flats:
        flat_folder = os.path.join(base_folder, 'Flats')
        create_master_flats_for_filters(flat_folder, calibration_folder, master_bias)

    # Placeholder for further processing of the light frames
    # Aligned frames, applying corrections, etc.

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

def normalize_frame(frame):
    """
    Normalize a frame by dividing each pixel by the mean of the frame.

    Parameters:
    frame : numpy array
        The frame to normalize.

    Returns:
    numpy array : Normalized frame.
    """
    mean_value = np.mean(frame)
    normalized_frame = frame / np.where(mean_value == 0, 1, mean_value)
    
    return normalized_frame

def create_master_dark(dark_folder, calibration_folder, master_bias=None):
    """
    Create a master dark frame by subtracting the master bias (if provided),
    computing the median of all dark frames in the specified folder, and saving the result
    in the calibration folder.

    Parameters:
    dark_folder : str
        The path to the folder containing dark frames.
    calibration_folder : str
        The folder to save the master calibration files.
    master_bias : numpy array, optional
        The master bias frame to subtract from each dark frame.

    Returns:
    None
    """
    dark_files = [f for f in os.listdir(dark_folder) if f.endswith('.fit') or f.endswith('.fits')]
    if not dark_files:
        raise FileNotFoundError(f"No dark frames found in folder: {dark_folder}")
    
    dark_frames = []

    for dark_file in dark_files:
        dark_path = os.path.join(dark_folder, dark_file)
        with fits.open(dark_path) as hdul:
            dark_data = hdul[0].data.astype(np.float32)
            if master_bias is not None:
                dark_data -= master_bias  # Subtract master bias from each dark frame
            dark_frames.append(dark_data)

    master_dark = calculate_median_frame(dark_frames)

    master_dark_path = os.path.join(calibration_folder, 'master_dark.fit')
    fits.writeto(master_dark_path, master_dark, overwrite=True)
    print(f"Master dark saved to {master_dark_path}")

def create_master_bias(bias_folder, calibration_folder):
    """
    Create a master bias frame by computing the median of all bias frames
    in the specified folder, and saving the result in the calibration folder.

    Parameters:
    bias_folder : str
        The path to the folder containing bias frames.
    calibration_folder : str
        The folder to save the master calibration files.

    Returns:
    None
    """
    bias_files = [f for f in os.listdir(bias_folder) if f.endswith('.fit') or f.endswith('.fits')]
    if not bias_files:
        raise FileNotFoundError(f"No bias frames found in folder: {bias_folder}")
    
    bias_frames = []

    for bias_file in bias_files:
        bias_path = os.path.join(bias_folder, bias_file)
        with fits.open(bias_path) as hdul:
            bias_data = hdul[0].data.astype(np.float32)
            bias_frames.append(bias_data)

    master_bias = calculate_median_frame(bias_frames)

    master_bias_path = os.path.join(calibration_folder, 'master_bias.fit')
    fits.writeto(master_bias_path, master_bias, overwrite=True)
    print(f"Master bias saved to {master_bias_path}")

def create_master_flats_for_filters(flat_folder, calibration_folder, master_bias=None):
    """
    Create a separate master flat frame for each filter by subtracting the master bias (if provided),
    normalizing each flat frame, computing the median of the normalized frames, and saving the result
    in the calibration folder.

    Parameters:
    flat_folder : str
        The path to the folder containing flat frames (subfolders for each filter).
    calibration_folder : str
        The folder to save the master calibration files.
    master_bias : numpy array, optional
        The master bias frame to subtract from each flat frame.

    Returns:
    None
    """
    filters = [f for f in os.listdir(flat_folder) if os.path.isdir(os.path.join(flat_folder, f))]
    for filter_name in filters:
        filter_folder = os.path.join(flat_folder, filter_name)
        flat_files = [f for f in os.listdir(filter_folder) if f.endswith('.fit') or f.endswith('.fits')]

        if not flat_files:
            raise FileNotFoundError(f"No flat frames found in folder: {filter_folder}")

        flat_frames = []

        for flat_file in flat_files:
            flat_path = os.path.join(filter_folder, flat_file)
            with fits.open(flat_path) as hdul:
                flat_data = hdul[0].data.astype(np.float32)
                if master_bias is not None:
                    flat_data -= master_bias  # Subtract master bias from each flat frame
                normalized_flat = normalize_frame(flat_data)
                flat_frames.append(normalized_flat)

        master_flat = calculate_median_frame(flat_frames)

        master_flat_path = os.path.join(calibration_folder, f'master_flat_{filter_name}.fit')
        fits.writeto(master_flat_path, master_flat, overwrite=True)
        print(f"Master flat for {filter_name} saved to {master_flat_path}")

def load_master_frame(folder, master_filename):
    """
    Load a master FITS file (dark, bias, flat, etc.) from the given folder.

    Parameters:
    folder : str
        The folder where the master FITS file is stored.
    master_filename : str
        The name of the master FITS file (without extension).

    Returns:
    numpy array : Loaded master FITS data.
    """
    master_file = os.path.join(folder, f"{master_filename}.fit")

    if not os.path.exists(master_file):
        raise FileNotFoundError(f"{master_filename} file not found in {folder}.")
    
    with fits.open(master_file) as hdul:
        data = hdul[0].data
    return data
