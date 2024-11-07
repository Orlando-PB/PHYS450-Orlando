import os
import numpy as np
from astropy.io import fits
from utils import (
    sort_files_into_subfolders,
    calculate_median_frame,
    normalize_frame,
    load_master_frame,
)
from source_extraction import extract_sources

from align_and_stack import align_and_stack_images

def process_light_images(
    base_folder, output_folder, use_flats=True, use_darks=True, use_biases=True, stack=False
):
    total_files = 0  # Initialize total_files counter

    # First, sort files into subfolders
    sorted_categories = sort_files_into_subfolders(base_folder)

    # Create master dark, bias, and flat frames (if needed)
    calibration_folder = os.path.join(output_folder, "calibration")
    os.makedirs(calibration_folder, exist_ok=True)

    master_dark = master_bias = master_flat = None

    if use_biases:
        bias_folder = os.path.join(base_folder, "Bias")
        create_master_bias(bias_folder, calibration_folder)
        master_bias = load_master_frame(calibration_folder, "master_bias")

    if use_darks:
        dark_folder = os.path.join(base_folder, "Darks")
        create_master_dark(dark_folder, calibration_folder, master_bias)
        master_dark = load_master_frame(calibration_folder, "master_dark")

    if use_flats:
        flat_folder = os.path.join(base_folder, "Flats")
        create_master_flats_for_filters(flat_folder, calibration_folder, master_bias)

    # Path to the main lights folder
    light_folder = os.path.join(base_folder, "Lights")
    calibrated_folder = os.path.join(output_folder, "calibrated")
    os.makedirs(calibrated_folder, exist_ok=True)

    # Loop over each filter found in the 'Lights' category
    for filter_name, light_files in sorted_categories["Light"].items():
        print(f"Processing filter: {filter_name}")

        # Create a subfolder in the calibrated directory for this filter
        filter_calibrated_folder = os.path.join(calibrated_folder, filter_name)
        os.makedirs(filter_calibrated_folder, exist_ok=True)

        # Loop over each light file for the current filter
        for light_file in light_files:
            try:
                # Construct the path to the current light frame
                light_path = os.path.join(light_folder, filter_name, light_file)
                print(f"Processing light frame: {light_path}")

                # Open the light frame FITS file
                with fits.open(light_path) as hdul:
                    light_data = hdul[0].data.astype(np.float32)

                # Subtract the master bias if available
                if master_bias is not None:
                    light_data -= master_bias
                    light_data = np.clip(light_data, 0, None)  # Ensure no negative values

                # Subtract the master dark if available
                if master_dark is not None:
                    light_data -= master_dark
                    light_data = np.clip(light_data, 0, None)  # Ensure no negative values

                # Apply the master flat if available
                if use_flats:
                    master_flat = load_master_frame(calibration_folder, f"master_flat_{filter_name}")
                    if master_flat is not None:
                        light_data /= master_flat

                # Save the calibrated light frame to the calibrated subfolder
                calibrated_light_path = os.path.join(
                    filter_calibrated_folder, f"calibrated_{light_file}"
                )
                fits.writeto(
                    calibrated_light_path, light_data, hdul[0].header, overwrite=True
                )
                print(f"Calibrated light frame saved to {calibrated_light_path}")

                # Increment total_files
                total_files += 1

            except Exception as e:
                print(f"Error processing {light_file}: {e}")

    # Stacking and LRGB combination
    if stack:
        stacked_folder = os.path.join(output_folder, "stacked")
        os.makedirs(stacked_folder, exist_ok=True)

        for filter_name, light_files in sorted_categories["Light"].items():
            print(f"Extracting stars for filter: {filter_name}")

            # Extract star positions after all images for this filter are calibrated
            for light_file in light_files:
                calibrated_light_path = os.path.join(
                    calibrated_folder, filter_name, f"calibrated_{light_file}"
                )
                extract_sources(calibrated_light_path)


            # Align and stack images after extracting star positions
            filter_output_folder = os.path.join(stacked_folder, filter_name)
            os.makedirs(filter_output_folder, exist_ok=True)
            align_and_stack_images(
                os.path.join(calibrated_folder, filter_name), filter_output_folder
            )

    return total_files  # Return the total number of images processed

def create_master_dark(dark_folder, calibration_folder, master_bias=None):
    dark_files = [
        f for f in os.listdir(dark_folder) if f.endswith(".fit") or f.endswith(".fits")
    ]
    if not dark_files:
        raise FileNotFoundError(f"No dark frames found in folder: {dark_folder}")

    dark_frames = []

    for dark_file in dark_files:
        dark_path = os.path.join(dark_folder, dark_file)
        with fits.open(dark_path) as hdul:
            dark_data = hdul[0].data.astype(np.float32)
            if master_bias is not None:
                dark_data -= master_bias  # Subtract master bias from each dark frame
                dark_data = np.clip(dark_data, 0, None)  # Clip negative values to 0
            dark_frames.append(dark_data)

    # Compute the median of the clipped dark frames
    master_dark = calculate_median_frame(dark_frames)
    master_dark = np.clip(
        master_dark, 0, None
    )  # Ensure no negative values in the final master dark

    # Save the master dark frame
    master_dark_path = os.path.join(calibration_folder, "master_dark.fit")
    fits.writeto(master_dark_path, master_dark, overwrite=True)
    print(f"Master dark saved to {master_dark_path}")

def create_master_bias(bias_folder, calibration_folder):
    bias_files = [
        f for f in os.listdir(bias_folder) if f.endswith(".fit") or f.endswith(".fits")
    ]
    if not bias_files:
        raise FileNotFoundError(f"No bias frames found in folder: {bias_folder}")

    bias_frames = []

    for bias_file in bias_files:
        bias_path = os.path.join(bias_folder, bias_file)
        with fits.open(bias_path) as hdul:
            bias_data = hdul[0].data.astype(np.float32)
            bias_frames.append(bias_data)

    master_bias = calculate_median_frame(bias_frames)

    master_bias_path = os.path.join(calibration_folder, "master_bias.fit")
    fits.writeto(master_bias_path, master_bias, overwrite=True)
    print(f"Master bias saved to {master_bias_path}")

def create_master_flats_for_filters(flat_folder, calibration_folder, master_bias=None):
    filters = [
        f for f in os.listdir(flat_folder) if os.path.isdir(os.path.join(flat_folder, f))
    ]
    for filter_name in filters:
        filter_folder = os.path.join(flat_folder, filter_name)
        flat_files = [
            f
            for f in os.listdir(filter_folder)
            if f.endswith(".fit") or f.endswith(".fits")
        ]

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

        master_flat_path = os.path.join(calibration_folder, f"master_flat_{filter_name}.fit")
        fits.writeto(master_flat_path, master_flat, overwrite=True)
        print(f"Master flat for {filter_name} saved to {master_flat_path}")
