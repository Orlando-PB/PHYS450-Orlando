#!/usr/bin/env python3
"""
photometric_calibration.py

This script loads an image’s JSON file (which was updated by the source extraction process)
and updates it with a calibrated magnitude (here referred to as "absolute magnitude")
for each source based on a calibration derived from Gaia reference stars.

Usage:
    python photometric_calibration.py <path_to_image_json>
"""

import sys
import json
import numpy as np
import os

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astroquery.gaia import Gaia

# Global debug flag
DEBUG = True

def query_gaia_catalog(ra_center, dec_center, radius_deg):
    """
    Query the Gaia catalogue around a central position.
    
    Parameters
    ----------
    ra_center, dec_center : float
        The central coordinates (in degrees) of the image.
    radius_deg : float
        The radius (in degrees) around the center to query.
    
    Returns
    -------
    gaia_results : astropy Table
        The table of Gaia sources.
    """
    coord = SkyCoord(ra=ra_center*u.deg, dec=dec_center*u.deg)
    print(f"Querying Gaia within {radius_deg} deg of RA={ra_center}, Dec={dec_center} ...")
    
    job = Gaia.cone_search_async(coordinate=coord, radius=radius_deg*u.deg)
    gaia_results = job.get_results()
    print(f"Found {len(gaia_results)} Gaia sources.")
    
    if DEBUG:
        # Print first few Gaia source positions for inspection.
        print("Sample Gaia source coordinates (RA, Dec):")
        for i in range(min(5, len(gaia_results))):
            print(f"  {i}: ({gaia_results['ra'][i]:.5f}, {gaia_results['dec'][i]:.5f})")
    
    return gaia_results

def cross_match_sources(sources, gaia_table, tolerance=2.0):
    """
    Cross-match extracted sources with Gaia stars.
    
    Parameters
    ----------
    sources : list of dict
        List of sources from the JSON file. Each should have 'ra', 'dec', and 'flux'.
    gaia_table : astropy Table
        The Gaia reference table (must have 'ra', 'dec', and a photometric column, e.g., 'phot_g_mean_mag').
    tolerance : float, optional
        Matching tolerance in arcseconds.
    
    Returns
    -------
    match_info : list of tuples
        Each tuple is (source_index, gaia_mag, inst_mag_diff)
        where inst_mag_diff = (Gaia_mag - instrumental_mag).
    """
    if not sources:
        print("No extracted sources available for matching.")
        return []

    # Build SkyCoord objects for our extracted sources and Gaia sources.
    try:
        source_ra = [src['ra'] for src in sources]
        source_dec = [src['dec'] for src in sources]
    except KeyError:
        print("One or more sources do not have 'ra' and 'dec' keys. Check your JSON output.")
        return []
    
    if DEBUG:
        print(f"Found {len(source_ra)} extracted sources. Sample positions:")
        for i in range(min(5, len(source_ra))):
            print(f"  Source {i}: RA={source_ra[i]:.5f}, Dec={source_dec[i]:.5f}")
    
    source_coords = SkyCoord(ra=source_ra*u.deg,
                             dec=source_dec*u.deg)
    gaia_coords = SkyCoord(ra=gaia_table['ra'], dec=gaia_table['dec'], unit='deg')
    
    # Perform the matching.
    idx, d2d, _ = source_coords.match_to_catalog_sky(gaia_coords)
    
    match_info = []
    for i, sep in enumerate(d2d):
        sep_arcsec = sep.arcsec
        if DEBUG:
            # Print each source's nearest neighbor separation even if it misses the tolerance.
            print(f"Source {i}: Separation to nearest Gaia star = {sep_arcsec:.2f} arcsec")
        if sep_arcsec < tolerance:
            gaia_mag = gaia_table['phot_g_mean_mag'][idx[i]]
            flux = sources[i].get('flux', None)
            if flux is None or flux <= 0:
                if DEBUG:
                    print(f"  Source {i} skipped due to invalid flux: {flux}")
                continue  # skip if no valid flux
            inst_mag = -2.5 * np.log10(flux)
            diff = gaia_mag - inst_mag
            match_info.append( (i, gaia_mag, diff) )
    
    print(f"Matched {len(match_info)} extracted sources with Gaia within {tolerance} arcsec tolerance.")
    if len(match_info) == 0:
        print("No matches found. Consider:")
        print("  - Verifying that your extracted source coordinates are in RA/Dec (not pixel values).")
        print("  - Increasing the matching tolerance (currently set to 2 arcsec).")
        print("  - Excluding extended or non-stellar sources (e.g., sources in the galaxy).")
    return match_info

def calibrate_sources(sources, match_info):
    """
    Compute the zero-point offset and update each source with a calibrated magnitude.
    
    Parameters
    ----------
    sources : list of dict
        List of sources to update.
    match_info : list of tuples (source_index, gaia_mag, diff)
    
    Returns
    -------
    zero_point : float
        The computed calibration zero point.
    """
    if not match_info:
        raise ValueError("No calibration stars found to compute zero point.")
    
    diffs = [info[2] for info in match_info]
    zero_point = np.median(diffs)
    print(f"Computed zero point offset: {zero_point:.3f} mag")
    
    # Update each source with the calibrated magnitude.
    for i, src in enumerate(sources):
        flux = src.get('flux', None)
        if flux is not None and flux > 0:
            inst_mag = -2.5 * np.log10(flux)
            calibrated_mag = inst_mag + zero_point
            src['absolute_magnitude'] = calibrated_mag
        else:
            src['absolute_magnitude'] = None
    return zero_point

def update_json_with_calibration(json_file, radius_deg=None):
    """
    Load the JSON file, perform photometric calibration using Gaia, and update the JSON with calibrated magnitudes.
    
    Parameters
    ----------
    json_file : str
        Path to the image JSON file.
    radius_deg : float, optional
        Radius (in degrees) to use for querying Gaia. If not provided, it will try to use the "Field Radius" key in the JSON.
    """
    if not os.path.exists(json_file):
        print(f"JSON file {json_file} does not exist.")
        return
    
    with open(json_file, "r") as infile:
        data = json.load(infile)
    
    # Check that the JSON file has the necessary WCS/image info.
    try:
        image_ra = data["Right Ascension"]
        image_dec = data["Declination"]
    except KeyError:
        print("Image center coordinates (Right Ascension and Declination) not found in JSON.")
        return
    
    # Determine the search radius for Gaia.
    if radius_deg is None:
        radius_deg = data.get("Field Radius", 0.5)  # default to 0.5 deg if not specified
    
    # Get the list of sources.
    sources = data.get("sources", [])
    if not sources:
        print("No sources found in JSON to calibrate.")
        return
    
    # OPTIONAL: Exclude sources that may be extended (for example, in a galaxy)
    # if your JSON includes parameters (e.g., "FWHM", "ellipticity", etc.) to flag extended sources.
    # For example:
    # sources = [src for src in sources if src.get('FWHM', 0) < some_threshold]
    
    # Query Gaia for reference stars.
    gaia_table = query_gaia_catalog(image_ra, image_dec, radius_deg)
    if len(gaia_table) == 0:
        print("No Gaia sources returned from query; cannot perform calibration.")
        return
    
    # Cross-match the extracted sources to Gaia.
    match_info = cross_match_sources(sources, gaia_table, tolerance=2.0)
    if not match_info:
        print("No matching Gaia stars found; cannot perform calibration.")
        return
    
    # Compute the zero point and update each source.
    zero_point = calibrate_sources(sources, match_info)
    
    # Add the zero point to the top-level JSON.
    data["photometric_zero_point"] = zero_point
    
    # Save the updated JSON.
    with open(json_file, "w") as outfile:
        json.dump(data, outfile, indent=4)
    print(f"Updated JSON file saved: {json_file}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python photometric_calibration.py <path_to_image_json>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    update_json_with_calibration(json_path)
