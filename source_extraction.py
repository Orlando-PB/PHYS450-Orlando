#!/usr/bin/env python3
"""
source_extraction.py

Extract sources from a FITS image using DAOStarFinder. If an astrometry JSON file
(with keys such as "Right Ascension", "Declination", "Pixel Scale (arcsec/pixel)",
and "Orientation (degrees)") exists, it will be used to build the WCS.
Otherwise, the WCS is taken from the FITS header.

Usage:
    sources = extract_sources(image_path, fwhm=3.0, threshold=5.0, astrometry_json_path="path/to/astrometry_solution.json")
"""

import os
import json
import math
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from photutils.detection import DAOStarFinder  # Updated import path

def build_wcs_from_astrometry(astrometry, nx, ny):
    """
    Build a simple TAN WCS using astrometry parameters from the JSON file.

    Parameters:
        astrometry : dict
            Must contain:
                "Right Ascension" (center RA in degrees),
                "Declination" (center Dec in degrees),
                "Pixel Scale (arcsec/pixel)" (float),
                "Orientation (degrees)" (float).
        nx, ny : int
            Dimensions of the image in pixels.

    Returns:
        wcs : astropy.wcs.WCS
            A WCS object built from the provided astrometry.
    """
    wcs = WCS(naxis=2)
    
    # Set the world coordinate (CRVAL) at the image center.
    wcs.wcs.crval = [astrometry["Right Ascension"], astrometry["Declination"]]
    wcs.wcs.crpix = [nx / 2.0, ny / 2.0]
    
    # Convert pixel scale from arcsec/pixel to deg/pixel.
    scale = astrometry["Pixel Scale (arcsec/pixel)"] / 3600.0
    
    # Orientation angle (in radians)
    theta = math.radians(astrometry["Orientation (degrees)"])
    
    # Construct the CD matrix.
    # Adjust the sign as needed for your system.
    cd11 = -scale * math.cos(theta)
    cd12 = scale * math.sin(theta)
    cd21 = scale * math.sin(theta)
    cd22 = scale * math.cos(theta)
    wcs.wcs.cd = [[cd11, cd12], [cd21, cd22]]
    
    # Set the projection types.
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    
    return wcs

def extract_sources(image_path, fwhm=3.0, threshold=5.0, astrometry_json_path=None):
    """
    Extract sources from a FITS image using DAOStarFinder.
    
    Parameters:
        image_path : str
            Path to the FITS image file.
        fwhm : float
            The full width at half maximum for the stars (default 3.0).
        threshold : float
            The detection threshold in sigma above the background (default 5.0).
        astrometry_json_path : str, optional
            Path to a JSON file containing astrometry parameters. If provided and found,
            its astrometry will be used to build the WCS.
    
    Returns:
        sources : list of dict
            A list of dictionaries, each containing parameters for one detected source.
            Typical keys include 'id', 'xcentroid', 'ycentroid', 'sharpness', 'roundness1',
            'roundness2', 'flux', plus added keys 'ra', 'dec', and 'count'.
    """
    # Open the FITS file and extract the image data and header.
    with fits.open(image_path) as hdul:
        data = hdul[0].data.astype(float)
        header = hdul[0].header
        ny, nx = data.shape  # image dimensions

    # Determine which WCS to use.
    if astrometry_json_path is not None and os.path.exists(astrometry_json_path):
        with open(astrometry_json_path, "r") as infile:
            astrometry_data = json.load(infile)
        wcs = build_wcs_from_astrometry(astrometry_data, nx, ny)
    else:
        wcs = WCS(header)

    # Estimate the background statistics using sigma-clipped statistics.
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)

    # Create a DAOStarFinder object using the specified fwhm and threshold * background std.
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * std)
    sources_table = daofind(data - median)

    # If no sources are detected, return an empty list.
    if sources_table is None:
        print("No sources detected.")
        return []

    # Convert the Astropy Table to a list of dictionaries and add WCS coordinates.
    sources = []
    for row in sources_table:
        # Convert each column value to a Python float.
        source = {col: float(row[col]) for col in sources_table.colnames}
        
        # Extract the pixel coordinates.
        xpix = source['xcentroid']
        ypix = source['ycentroid']
        
        # Convert pixel coordinates to world coordinates using the chosen WCS.
        # Use origin=0 because DAOStarFinder returns 0-indexed coordinates.
        world_coords = wcs.wcs_pix2world([[xpix, ypix]], 0)[0]
        source['ra'] = float(world_coords[0])
        source['dec'] = float(world_coords[1])
        
        # Alias the flux as "count" (uncalibrated counts).
        if 'flux' in source:
            source['count'] = source['flux']
        
        sources.append(source)

    return sources
