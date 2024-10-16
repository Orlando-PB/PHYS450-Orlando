import os
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
import numpy as np

def get_wcs_from_header(file_path):
    """
    Extract the WCS (world coordinate system) from the FITS header.
    """
    with fits.open(file_path) as hdul:
        wcs = WCS(hdul[0].header)
    return wcs

def align_to_reference_frame(file_path, reference_wcs, shape_out):
    """
    Align a FITS image to a reference WCS frame.
    """
    with fits.open(file_path) as hdul:
        data = hdul[0].data
        wcs = WCS(hdul[0].header)
        # Reproject the image onto the reference WCS frame
        reprojected_data, _ = reproject_interp((data, wcs), reference_wcs, shape_out=shape_out)
    return reprojected_data

def align_light_frames(lights_folder):
    """
    Align all light frames found in the Lights folder and its subfolders to a reference WCS frame.
    The first light frame will be used as the reference.
    """
    light_files = []
    
    # Traverse through each filter folder inside the 'Lights' folder
    for root, _, files in os.walk(lights_folder):
        for file in files:
            if file.endswith('.fit'):
                light_files.append(os.path.join(root, file))

    if not light_files:
        raise FileNotFoundError("No light frames found for alignment.")

    reference_wcs = None
    aligned_frames = []

    # Use the first light frame as reference
    reference_frame = light_files[0]
    reference_wcs = get_wcs_from_header(reference_frame)
    with fits.open(reference_frame) as hdul:
        shape_out = hdul[0].data.shape

    # Align each light frame to the reference WCS
    for file_path in light_files:
        aligned_data = align_to_reference_frame(file_path, reference_wcs, shape_out)
        aligned_frames.append(aligned_data)

    return aligned_frames, reference_wcs

def save_aligned_frames(aligned_frames, base_folder):
    """
    Save the aligned frames as new FITS files.
    """
    output_folder = os.path.join(base_folder, "aligned")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, data in enumerate(aligned_frames):
        output_file = os.path.join(output_folder, f"aligned_{i+1}.fit")
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(output_file, overwrite=True)
        print(f"Saved aligned frame: {output_file}")

    print(f"All aligned frames saved to {output_folder}")
