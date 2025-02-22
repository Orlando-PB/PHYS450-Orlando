
import os
import json
import math
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from photutils.detection import DAOStarFinder 

def build_wcs_from_astrometry(astrometry, nx, ny):

    wcs = WCS(naxis=2)
    
    wcs.wcs.crval = [astrometry["Right Ascension"], astrometry["Declination"]]
    wcs.wcs.crpix = [nx / 2.0, ny / 2.0]
    
    scale = astrometry["Pixel Scale (arcsec/pixel)"] / 3600.0
    
    theta = math.radians(astrometry["Orientation (degrees)"])
    
    cd11 = -scale * math.cos(theta)
    cd12 = scale * math.sin(theta)
    cd21 = scale * math.sin(theta)
    cd22 = scale * math.cos(theta)
    wcs.wcs.cd = [[cd11, cd12], [cd21, cd22]]
    
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    
    return wcs

def extract_sources(image_path, fwhm=3.0, threshold=5.0, astrometry_json_path=None):

    with fits.open(image_path) as hdul:
        data = hdul[0].data.astype(float)
        header = hdul[0].header
        ny, nx = data.shape 

    if astrometry_json_path is not None and os.path.exists(astrometry_json_path):
        with open(astrometry_json_path, "r") as infile:
            astrometry_data = json.load(infile)
        wcs = build_wcs_from_astrometry(astrometry_data, nx, ny)
    else:
        wcs = WCS(header)

    mean, median, std = sigma_clipped_stats(data, sigma=3.0)

    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * std)
    sources_table = daofind(data - median)

    if sources_table is None:
        print("No sources detected.")
        return []

    sources = []
    for row in sources_table:
        source = {col: float(row[col]) for col in sources_table.colnames}
        
        xpix = source['xcentroid']
        ypix = source['ycentroid']
        
        world_coords = wcs.wcs_pix2world([[xpix, ypix]], 0)[0]
        source['ra'] = float(world_coords[0])
        source['dec'] = float(world_coords[1])
        
        if 'flux' in source:
            source['count'] = source['flux']
        
        sources.append(source)

    return sources
