# fits_processor.py
import os
import json
import numpy as np
from astropy.io import fits
import astrometry
from utils import sort_files_into_subfolders
from source_extraction import extract_sources
from photometric_calibration import update_json_with_calibration  # New import

def process_light_images(base_folder, output_folder,
                         use_flats=True, use_darks=True, use_biases=True,
                         do_astrometry=True, astrometry_api_key=None):
    """
    Process light images: bias, dark, flat calibration. Optionally run astrometry, source extraction,
    and photometric calibration after calibration, updating the resulting JSON file (saved next to each image).

    Parameters:
    -----------
    base_folder : str
        The base folder containing raw images (subfolders: 'Lights', 'Darks', 'Bias', 'Flats').
    output_folder : str
        The output folder to save calibration files and calibrated images.
    use_flats, use_darks, use_biases : bool
        Flags to enable/disable usage of certain calibration frames.
    do_astrometry : bool
        Whether to perform astrometry on each calibrated image.
    astrometry_api_key : str
        The astrometry.net API key (required if do_astrometry is True).
    """
    # First, sort files into subfolders
    sorted_categories = sort_files_into_subfolders(base_folder)

    # Create directory for calibration output
    calibration_folder = os.path.join(output_folder, 'calibration')
    os.makedirs(calibration_folder, exist_ok=True)

    # Master frames
    master_dark = master_bias = master_flat = None

    # ------------------------------------------------
    # Create Master Bias
    if use_biases:
        bias_folder = os.path.join(base_folder, 'Bias')
        create_master_bias(bias_folder, calibration_folder)
        master_bias = load_master_frame(calibration_folder, "master_bias")

    # ------------------------------------------------
    # Create Master Dark
    if use_darks:
        dark_folder = os.path.join(base_folder, 'Darks')
        create_master_dark(dark_folder, calibration_folder, master_bias)
        master_dark = load_master_frame(calibration_folder, "master_dark")

    # ------------------------------------------------
    # Create Master Flats (separately for each filter)
    if use_flats:
        flat_folder = os.path.join(base_folder, 'Flats')
        create_master_flats_for_filters(flat_folder, calibration_folder, master_bias)

    # ------------------------------------------------
    # Setup the astrometry session if requested
    if do_astrometry:
        try:
            # Override the default API key if one is provided
            if astrometry_api_key:
                astrometry.API_KEY = astrometry_api_key
            astrometry_session = astrometry.setup_astrometry()
        except Exception as e:
            print("Error setting up astrometry:", e)
            astrometry_session = None
    else:
        astrometry_session = None

    # ------------------------------------------------
    # Calibrate Light Images
    light_folder = os.path.join(base_folder, 'Lights')
    calibrated_folder = os.path.join(output_folder, 'calibrated')
    os.makedirs(calibrated_folder, exist_ok=True)

    for filter_name, light_files in sorted_categories['Light'].items():
        print(f"Processing filter: {filter_name}")

        # Create a subfolder in the calibrated directory for this filter
        filter_calibrated_folder = os.path.join(calibrated_folder, filter_name)
        os.makedirs(filter_calibrated_folder, exist_ok=True)

        # Loop over each light file for the current filter
        for light_file in light_files:
            try:
                light_path = os.path.join(light_folder, filter_name, light_file)
                print(f"Processing light frame: {light_path}")

                # Open the light frame FITS file and extract header for later use
                with fits.open(light_path) as hdul:
                    light_data = hdul[0].data.astype(np.float32)
                    header = hdul[0].header

                # Subtract Master Bias
                if master_bias is not None:
                    light_data -= master_bias
                    light_data = np.clip(light_data, 0, None)  # Ensure no negative values

                # Subtract Master Dark
                if master_dark is not None:
                    light_data -= master_dark
                    light_data = np.clip(light_data, 0, None)  # Ensure no negative values

                # Apply Master Flat
                if use_flats:
                    master_flat = load_master_frame(calibration_folder, f"master_flat_{filter_name}")
                    if master_flat is not None:
                        light_data /= master_flat

                # Save the calibrated light frame
                calibrated_light_path = os.path.join(filter_calibrated_folder, f"calibrated_{light_file}")
                fits.writeto(calibrated_light_path, light_data, header, overwrite=True)
                print(f"Calibrated light frame saved to {calibrated_light_path}")

                # ------------------------------------------------
                # Run astrometry on the calibrated image if requested
                if do_astrometry and astrometry_session:
                    try:
                        print("Running astrometry on calibrated image:", calibrated_light_path)
                        astrometry_result = astrometry.process_image(calibrated_light_path, astrometry_session)
                        print("Astrometry result:", astrometry_result)
                    except Exception as e:
                        print(f"Error during astrometry for {calibrated_light_path}: {e}")

                # ------------------------------------------------
                # Run source extraction on the calibrated image and update the JSON file
                try:
                    print("Running source extraction on calibrated image:", calibrated_light_path)
                    sources = extract_sources(calibrated_light_path)

                    # Determine the JSON file path (same naming as in astrometry module)
                    image_dir = os.path.dirname(calibrated_light_path)
                    base_name = os.path.splitext(os.path.basename(calibrated_light_path))[0]
                    json_filename = os.path.join(image_dir, f"{base_name}_astrometry_solution.json")

                    # If the JSON file exists, load it; otherwise, start with an empty dictionary.
                    if os.path.exists(json_filename):
                        with open(json_filename, "r") as infile:
                            data = json.load(infile)
                    else:
                        data = {}

                    # Update the JSON data with the sources
                    data["sources"] = sources

                    with open(json_filename, "w") as outfile:
                        json.dump(data, outfile, indent=4)
                    print("Updated JSON file with source extraction data:", json_filename)
                except Exception as e:
                    print(f"Error during source extraction for {calibrated_light_path}: {e}")

                # ------------------------------------------------
                # Run photometric calibration on the JSON file
                try:
                    print("Running photometric calibration on JSON file:", json_filename)
                    update_json_with_calibration(json_filename)
                except Exception as e:
                    print(f"Error during photometric calibration for {calibrated_light_path}: {e}")

            except Exception as e:
                print(f"Error processing {light_file}: {e}")

# ----------------------------------------------------------------------------------------
# Below: Unchanged helper functions for bias, dark, and flat creation (with minor edits).
# ----------------------------------------------------------------------------------------

def calculate_median_frame(frames):
    """
    Calculate the median of a list of FITS frames.
    """
    stacked_frames = np.stack(frames, axis=0)
    median_frame = np.median(stacked_frames, axis=0)
    return median_frame

def normalize_frame(frame):
    """
    Normalize a frame by dividing each pixel by the mean of the frame.
    """
    mean_value = np.mean(frame)
    normalized_frame = frame / np.where(mean_value == 0, 1, mean_value)
    return normalized_frame

def create_master_dark(dark_folder, calibration_folder, master_bias=None):
    """
    Create a master dark frame by subtracting the master bias (if provided),
    computing the median of all dark frames, and saving the result.
    """
    dark_files = [f for f in os.listdir(dark_folder) if f.lower().endswith(('.fit', '.fits'))]
    if not dark_files:
        raise FileNotFoundError(f"No dark frames found in folder: {dark_folder}")

    dark_frames = []
    for dark_file in dark_files:
        dark_path = os.path.join(dark_folder, dark_file)
        with fits.open(dark_path) as hdul:
            dark_data = hdul[0].data.astype(np.float32)
            if master_bias is not None:
                dark_data -= master_bias
                dark_data = np.clip(dark_data, 0, None)
            dark_frames.append(dark_data)

    master_dark = calculate_median_frame(dark_frames)
    master_dark = np.clip(master_dark, 0, None)

    master_dark_path = os.path.join(calibration_folder, 'master_dark.fit')
    fits.writeto(master_dark_path, master_dark, overwrite=True)
    print(f"Master dark saved to {master_dark_path}")

def create_master_bias(bias_folder, calibration_folder):
    """
    Create a master bias frame by computing the median of all bias frames.
    """
    bias_files = [f for f in os.listdir(bias_folder) if f.lower().endswith(('.fit', '.fits'))]
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
    Create a separate master flat frame for each filter subfolder.
    """
    filters = [f for f in os.listdir(flat_folder) if os.path.isdir(os.path.join(flat_folder, f))]
    for filter_name in filters:
        filter_path = os.path.join(flat_folder, filter_name)
        flat_files = [f for f in os.listdir(filter_path) if f.lower().endswith(('.fit', '.fits'))]
        if not flat_files:
            raise FileNotFoundError(f"No flat frames found in {filter_path}")

        flat_frames = []
        for flat_file in flat_files:
            flat_path = os.path.join(filter_path, flat_file)
            with fits.open(flat_path) as hdul:
                flat_data = hdul[0].data.astype(np.float32)
                if master_bias is not None:
                    flat_data -= master_bias
                normalized_flat = normalize_frame(flat_data)
                flat_frames.append(normalized_flat)

        master_flat = calculate_median_frame(flat_frames)
        master_flat_path = os.path.join(calibration_folder, f"master_flat_{filter_name}.fit")
        fits.writeto(master_flat_path, master_flat, overwrite=True)
        print(f"Master flat for {filter_name} saved to {master_flat_path}")

def load_master_frame(folder, master_filename):
    """
    Load a master FITS file (dark, bias, flat, etc.) from the given folder.
    """
    master_file = os.path.join(folder, f"{master_filename}.fit")
    if not os.path.exists(master_file):
        raise FileNotFoundError(f"{master_filename}.fit file not found in {folder}.")

    with fits.open(master_file) as hdul:
        data = hdul[0].data
    return data
