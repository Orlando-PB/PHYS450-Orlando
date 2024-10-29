import numpy as np
from astropy.io import fits
from photutils.detection import DAOStarFinder
from astropy.stats import mad_std
from scipy.spatial import KDTree
import os

def extract_bright_stars(fits_file, fwhm=3.0, threshold=5.0, min_distance=10.0):
    """
    Extract positions of bright stars from a FITS file using DAOStarFinder, filtering out stars too close to others,
    and save them to a text file.

    Parameters:
    fits_file : str
        Path to the FITS file to analyze.
    fwhm : float, optional
        Full Width at Half Maximum of stars in pixels (default is 3.0).
    threshold : float, optional
        Detection threshold for stars (in terms of standard deviation, default is 5.0).
    min_distance : float, optional
        Minimum distance between stars to consider them as isolated (default is 10.0 pixels).

    Returns:
    None
    """
    try:
        # Open the FITS file and extract image data
        with fits.open(fits_file) as hdul:
            image_data = hdul[0].data.astype(np.float32)

        # Estimate the background noise using median absolute deviation
        bkg_sigma = mad_std(image_data)

        # Initialize DAOStarFinder
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * bkg_sigma)

        # Detect stars
        stars = daofind(image_data - np.median(image_data))

        if stars is None:
            print(f"No stars found in {fits_file}")
            return

        # Extract the x and y positions of the stars
        star_positions = np.array([stars['xcentroid'], stars['ycentroid']]).T

        # Use KDTree to find pairs of stars within min_distance
        tree = KDTree(star_positions)
        isolated_positions = []

        for i, position in enumerate(star_positions):
            distances, indices = tree.query(position, k=2)
            # distances[1] is the distance to the nearest neighbor (excluding itself at distance 0)
            if distances[1] >= min_distance:
                isolated_positions.append(position)

        # Define the output file name (same folder, same name but with "_star_positions.txt")
        output_file = fits_file.replace('.fits', '_star_positions.txt').replace('.fit', '_star_positions.txt')

        # Save the filtered star positions to a text file
        with open(output_file, 'w') as f:
            for x, y in isolated_positions:
                f.write(f"{x:.2f},{y:.2f}\n")

        print(f"Filtered star positions saved to {output_file}")

    except Exception as e:
        print(f"Error processing {fits_file}: {e}")
