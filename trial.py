import os
import numpy as np
from astropy.io import fits

# Define the base folder where all files are located
base_folder = "./demo"
output_folder = os.path.join(base_folder, "Output")

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to combine (average) FITS files based on a keyword in the filename
def load_combined_fits(keyword, filter_name=None):
    if filter_name:
        fits_files = [f for f in os.listdir(base_folder) if keyword in f and filter_name in f and f.endswith('.fit')]
    else:
        fits_files = [f for f in os.listdir(base_folder) if keyword in f and f.endswith('.fit')]

    if not fits_files:
        raise FileNotFoundError(f"No FITS files found for keyword '{keyword}' and filter '{filter_name}'")

    combined_data = None

    for file in fits_files:
        try:
            with fits.open(os.path.join(base_folder, file)) as hdul:
                data = hdul[0].data.astype(np.float32)  # Convert to float32 for precision
                if combined_data is None:
                    combined_data = data
                else:
                    combined_data += data
        except Exception as e:
            raise IOError(f"Error processing file '{file}': {e}")

    # Take the average (mean) of the stacked images
    combined_data = combined_data / len(fits_files)
    return combined_data

# Try loading the master dark and bias frames, raise errors if missing or incompatible
try:
    master_dark = load_combined_fits("Dark")
except FileNotFoundError as e:
    raise FileNotFoundError("Master dark frames are missing or not found.") from e

try:
    master_bias = load_combined_fits("Bias")
except FileNotFoundError as e:
    raise FileNotFoundError("Master bias frames are missing or not found.") from e

# Process each light (science) image
for file in os.listdir(base_folder):
    if "Light" in file and file.endswith('.fit'):  # Identify light frames by keyword "Light"
        # Extract the filter from the filename (e.g., "Light_M31_180.0s_Bin1_R_20231015-054317_0010.fit")
        try:
            parts = file.split("_")
            # Find the part that starts with "Bin", and the filter name comes after that
            bin_part = next(part for part in parts if part.startswith("Bin"))
            filter_name = parts[parts.index(bin_part) + 1]
        except (IndexError, StopIteration):
            raise ValueError(f"Cannot extract filter name from file '{file}'")

        # Load the corresponding flat frame for the specific filter
        try:
            master_flat = load_combined_fits("Flat", filter_name)
            # Normalize the master flat frame to prevent over/under correction
            master_flat = master_flat / np.median(master_flat)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing flat frames for filter '{filter_name}'.") from e
        except ZeroDivisionError:
            raise ValueError(f"Flat frame for filter '{filter_name}' has invalid or zero median value.")
        except Exception as e:
            raise IOError(f"Error loading or processing flat frames for filter '{filter_name}': {e}")

        # Load the light image
        light_file_path = os.path.join(base_folder, file)
        try:
            with fits.open(light_file_path) as hdul:
                light_data = hdul[0].data.astype(np.float32)  # Convert to float32 for precision
                header = hdul[0].header
        except Exception as e:
            raise IOError(f"Error loading light frame '{file}': {e}")

        # Calibrate the image: subtract bias and dark frames
        calibrated_data = light_data - (master_dark + master_bias)

        # Divide by the master flat field to correct for vignetting and pixel response variations
        calibrated_data = calibrated_data / master_flat

        # Ensure no negative values by clipping to a minimum of 0
        calibrated_data = np.clip(calibrated_data, 0, None)

        # Save the calibrated image to the output folder
        output_file_path = os.path.join(output_folder, file)
        try:
            fits.writeto(output_file_path, calibrated_data, header, overwrite=True)
        except Exception as e:
            raise IOError(f"Error saving calibrated image '{file}': {e}")

print(f"Calibrated images (with flat correction by filter) saved to {output_folder}")
