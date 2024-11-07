import numpy as np
from photutils.segmentation import detect_sources, deblend_sources, SourceCatalog
from photutils.background import Background2D, MedianBackground
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.convolution import Gaussian2DKernel

def extract_sources(fits_file, fwhm=3.0, threshold=5.0):
    """
    Detect galaxies and stars in a FITS file, mask galaxies, and extract stars with magnitudes.
    Save star positions and magnitudes, and galaxy positions and sizes to text files.
    """
    try:
        # Open the FITS file and extract image data
        with fits.open(fits_file) as hdul:
            image_data = hdul[0].data.astype(np.float32)
            header = hdul[0].header

        # Estimate the background and subtract it
        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = MedianBackground()
        bkg = Background2D(image_data, (50, 50), filter_size=(3, 3),
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        data_sub = image_data - bkg.background

        # Create a Gaussian kernel for source detection
        kernel = Gaussian2DKernel(fwhm / 2.355, x_size=3, y_size=3)
        kernel.normalize()

        # Detect all sources (galaxies and stars)
        threshold_value = threshold * bkg.background_rms_median
        segm = detect_sources(data_sub, threshold_value, npixels=5, filter_kernel=kernel)
        if segm is None:
            print(f"No sources found in {fits_file}")
            return

        # Deblend sources
        segm_deblend = deblend_sources(data_sub, segm, npixels=5,
                                       nlevels=32, contrast=0.001)
        if segm_deblend is None:
            print(f"No sources found after deblending in {fits_file}")
            return

        # Create a catalog of sources
        cat = SourceCatalog(data_sub, segm_deblend)
        tbl = cat.to_table()

        # Compute ellipticity manually
        semimajor_sigma = cat.semimajor_sigma.value
        semiminor_sigma = cat.semiminor_sigma.value
        ellipticity = 1.0 - (semiminor_sigma / semimajor_sigma)
        tbl['ellipticity'] = ellipticity

        # Classify sources as stars or galaxies based on morphological parameters
        source_area = tbl['area'].data  # Number of pixels in the source

        # Define thresholds to classify galaxies (adjust as needed)
        galaxy_ellipticity_threshold = 0.3
        galaxy_area_threshold = 50  # pixels

        galaxies_mask = (ellipticity > galaxy_ellipticity_threshold) | (source_area > galaxy_area_threshold)
        stars_mask = ~galaxies_mask

        galaxies = tbl[galaxies_mask]
        stars = tbl[stars_mask]

        print(f"Number of galaxies: {len(galaxies)}")
        print(f"Number of stars: {len(stars)}")

        # Save galaxy positions and sizes
        galaxy_positions = []
        for source in galaxies:
            x = source['xcentroid']
            y = source['    ']
            a = source['semimajor_sigma'] * 3  # Approximate size
            b = source['semiminor_sigma'] * 3
            theta = source['orientation'].to('deg').value
            galaxy_positions.append((x.value, y.value, a.value, b.value, theta))

        galaxy_output_file = fits_file.replace('.fits', '_galaxies.txt').replace('.fit', '_galaxies.txt')
        with open(galaxy_output_file, 'w') as f:
            for x, y, a, b, theta in galaxy_positions:
                f.write(f"{x:.2f},{y:.2f},{a:.2f},{b:.2f},{theta:.2f}\n")

        print(f"Galaxy positions saved to {galaxy_output_file}")

        # Create a mask for galaxies to exclude them from star detection
        galaxy_labels = galaxies['label']
        galaxy_mask = segm_deblend.copy()
        galaxy_mask.remove_labels(galaxy_labels)
        galaxy_mask = galaxy_mask.data.astype(bool)

        # Mask out the galaxies in the data
        data_for_stars = data_sub.copy()
        data_for_stars[~galaxy_mask] = np.nan  # Mask out galaxies

        # Now detect stars using the same method
        segm_stars = detect_sources(data_for_stars, threshold_value, npixels=5, filter_kernel=kernel)
        if segm_stars is None:
            print(f"No stars found in {fits_file}")
            return

        # Create a catalog for stars
        cat_stars = SourceCatalog(data_sub, segm_stars)
        tbl_stars = cat_stars.to_table()

        # Measure magnitudes (instrumental magnitudes)
        tbl_stars['mag'] = -2.5 * np.log10(tbl_stars['segment_flux'])

        # Extract star positions and magnitudes
        star_positions = []
        for source in tbl_stars:
            x = source['xcentroid']
            y = source['ycentroid']
            mag = source['mag']
            star_positions.append((x.value, y.value, mag))

        # Sort stars by magnitude (brightest first)
        star_positions.sort(key=lambda s: s[2])

        # Save star positions and magnitudes
        star_output_file = fits_file.replace('.fits', '_stars.txt').replace('.fit', '_stars.txt')
        with open(star_output_file, 'w') as f:
            for x, y, mag in star_positions:
                f.write(f"{x:.2f},{y:.2f},{mag:.2f}\n")

        print(f"Star positions and magnitudes saved to {star_output_file}")

    except Exception as e:
        print(f"Error processing {fits_file}: {e}")
