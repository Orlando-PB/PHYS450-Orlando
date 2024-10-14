import os
import numpy as np
from astropy.io import fits
from utils import load_combined_fits
from PIL import Image
from alignment import align_light_frames, save_aligned_frames

def process_light_images(base_folder, output_folder, use_flats=True, use_darks=True, use_biases=True, combine_lrgb=False, align_frames=True, progress_var=None, progress_bar=None):
    # Align the light frames first, if enabled
    if align_frames:
        aligned_frames, reference_wcs = align_light_frames(base_folder)
        save_aligned_frames(aligned_frames, base_folder)
        print(f"Light frames aligned to RA/DEC coordinates.")
    else:
        aligned_frames = None

    # Continue with the regular calibration workflow...
    master_dark = master_bias = None
    if use_darks:
        try:
            master_dark = load_combined_fits(base_folder, "Dark")
        except FileNotFoundError as e:
            raise FileNotFoundError("Master dark frames are missing or not found.") from e

    if use_biases:
        try:
            master_bias = load_combined_fits(base_folder, "Bias")
        except FileNotFoundError as e:
            raise FileNotFoundError("Master bias frames are missing or not found.") from e

    light_files = [f for f in os.listdir(base_folder) if "Light" in f and f.endswith('.fit')]
    total_files = len(light_files)

    # List to hold calibrated frames for stacking
    calibrated_frames = []

    # Process each light (science) image
    for i, file in enumerate(light_files):
        # Load and calibrate the aligned frames (if they exist)
        if aligned_frames is not None:
            calibrated_data = aligned_frames[i]
        else:
            light_file_path = os.path.join(base_folder, file)
            with fits.open(light_file_path) as hdul:
                light_data = hdul[0].data.astype(np.float32)
                calibrated_data = light_data

        # Apply dark, bias, and flat corrections (if toggled)
        if use_darks and master_dark is not None:
            calibrated_data -= master_dark
        if use_biases and master_bias is not None:
            calibrated_data -= master_bias
        if use_flats:
            # Assuming flats are handled earlier or can be added here
            pass
        
        # Store the calibrated frame for stacking
        calibrated_frames.append(calibrated_data)

        # Save the calibrated image
        output_file_path = os.path.join(output_folder, f"calibrated_{i+1}.fit")
        fits.writeto(output_file_path, calibrated_data, overwrite=True)
        print(f"Saved calibrated frame: {output_file_path}")

        # Update the progress bar, if available
        if progress_var and progress_bar:
            progress = (i + 1) / total_files * 100
            progress_var.set(progress)
            progress_bar.update()

    # After processing all frames, stack them
    if calibrated_frames:
        final_image = stack_frames(calibrated_frames)
        save_final_image(final_image, output_folder)
        print(f"Final stacked image saved to {output_folder}")

    return total_files

def stack_frames(frames, method="average"):
    """
    Stack the frames using the specified method (average, median).
    :param frames: List of 2D numpy arrays (calibrated light frames)
    :param method: Stacking method (default is average, can also use 'median')
    :return: Stacked 2D numpy array
    """
    stack = None
    if method == "average":
        stack = np.mean(frames, axis=0)
    elif method == "median":
        stack = np.median(frames, axis=0)
    return stack

def save_final_image(final_image, output_folder, filename="stacked_image.fit"):
    """
    Save the final stacked image as a FITS file.
    """
    output_file = os.path.join(output_folder, filename)
    hdu = fits.PrimaryHDU(final_image)
    hdu.writeto(output_file, overwrite=True)
    print(f"Final stacked image saved as: {output_file}")
