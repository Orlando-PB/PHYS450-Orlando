import os
import numpy as np
from astropy.io import fits
from scipy.ndimage import shift
from scipy.spatial import cKDTree

def align_and_stack_images(calibrated_folder, output_folder):
    """
    Align and stack calibrated images for a given filter using individual star position files.

    Parameters:
    calibrated_folder : str
        The folder containing calibrated light frames for a filter.
    output_folder : str
        The folder to save the stacked image.

    Returns:
    None
    """
    # Collect star position files and their corresponding image files
    star_positions = {}
    for file_name in os.listdir(calibrated_folder):
        if file_name.endswith('_star_positions.txt'):
            light_file = file_name.replace('_star_positions.txt', '.fit')
            star_positions[light_file] = os.path.join(calibrated_folder, file_name)

    # Ensure there are at least two files with stars to align
    if len(star_positions) < 2:
        raise ValueError("Not enough images with star positions to perform alignment")

    # Reference star positions from the first file
    reference_file = list(star_positions.keys())[0]
    reference_stars = load_star_positions(star_positions[reference_file])

    # List to store all aligned frames
    aligned_frames = []

    # Align all images to the reference
    for light_file, star_file in star_positions.items():
        stars = load_star_positions(star_file)

        # Find nearest matching pairs of stars
        matched_star_pairs = find_nearest_star_pairs(reference_stars, stars)

        # Calculate shifts based on matched star positions
        shifts = [(r[0] - s[0], r[1] - s[1]) for r, s in matched_star_pairs]
        avg_shift = np.mean(shifts, axis=0)

        light_path = os.path.join(calibrated_folder, light_file)
        if light_file == reference_file:
            # No need to shift the reference image
            with fits.open(light_path) as hdul:
                aligned_frames.append(hdul[0].data.astype(np.float32))
        else:
            # Apply the shift to the image
            with fits.open(light_path) as hdul:
                data = hdul[0].data.astype(np.float32)
                shifted_data = shift(data, shift=avg_shift)
                aligned_frames.append(shifted_data)

    if len(aligned_frames) < 2:
        raise ValueError("Not enough aligned frames to perform stacking")

    # Stack aligned frames by calculating the median
    stacked_frame = np.median(np.stack(aligned_frames, axis=0), axis=0)

    # Save the stacked frame
    stacked_file = os.path.join(output_folder, 'stacked_image.fits')
    fits.writeto(stacked_file, stacked_frame, overwrite=True)
    print(f"Stacked image saved to {stacked_file}")
    return stacked_frame

def load_star_positions(star_file):
    """
    Load star positions from a text file.

    Parameters:
    star_file : str
        Path to the text file containing star positions.

    Returns:
    List of tuples containing (x, y) positions of stars.
    """
    positions = []
    with open(star_file, 'r') as f:
        for line in f:
            if not line.strip().endswith(':'):  # Ignore file headers
                x, y = line.strip('()').split(',')
                positions.append((float(x), float(y)))
    return positions

def find_nearest_star_pairs(reference_stars, stars):
    """
    Find the nearest matching pairs of stars between reference and current stars.

    Parameters:
    reference_stars : list of tuples
        The (x, y) positions of stars in the reference image.
    stars : list of tuples
        The (x, y) positions of stars in the current image.

    Returns:
    List of tuples with matched (reference_star, current_star) pairs.
    """
    if len(reference_stars) == 0 or len(stars) == 0:
        raise ValueError("No stars detected in one of the images.")

    ref_tree = cKDTree(reference_stars)
    matched_pairs = []
    for star in stars:
        dist, idx = ref_tree.query(star)
        matched_pairs.append((reference_stars[idx], star))

    return matched_pairs
