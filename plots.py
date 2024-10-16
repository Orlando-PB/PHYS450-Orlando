import os
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np

def show_fits_info(fits_file):
    """
    Opens a FITS file, shows its statistics (mean, min, max, median, and total pixel count),
    and plots a histogram of pixel values with the statistics annotated on the plot.

    Parameters:
    fits_file (str): Path to the FITS file to analyze.
    """
    try:
        # Open the FITS file
        with fits.open(fits_file) as hdul:
            data = hdul[0].data  # Assuming the data is in the primary HDU
            
            if data is None:
                print(f"No data found in {fits_file}")
                return
            
            # Calculate statistics
            mean_val = np.mean(data)
            min_val = np.min(data)
            max_val = np.max(data)
            median_val = np.median(data)
            total_val = np.sum(data)  # Calculate the total pixel value

            # Plot histogram
            plt.figure()
            plt.hist(data.flatten(), bins=500, color='blue', alpha=0.7)
            plt.yscale('linear')  # Set y-axis to linear scale
            plt.title(f'Histogram of FITS File: {os.path.basename(fits_file)}')
            plt.xlabel('Pixel Value')
            plt.ylabel('Count')

            # Display statistics on the plot
            textstr = (f'Total: {total_val:.2f}\n'
                       f'Mean: {mean_val:.2f}\n'
                       f'Median: {median_val:.2f}\n'
                       f'Min: {min_val:.2f}\n'
                       f'Max: {max_val:.2f}')
                       
            
            # Add a text box with the statistics in the upper right corner of the plot
            plt.gcf().text(0.95, 0.95, textstr, fontsize=12, verticalalignment='top', 
                           horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

            # Show the plot
            plt.show()
    
    except Exception as e:
        print(f"Error processing {fits_file}: {e}")


def plot_histograms_for_master_flats(calibration_folder, output_folder):
    """
    Plots histograms for master flats and saves the histograms as PNG files.

    Parameters:
    calibration_folder (str): Path to the folder containing master flats (e.g., output/calibration).
    output_folder (str): Path to the output folder where the histograms will be saved.
    """

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Ensure the calibration folder exists
    if not os.path.exists(calibration_folder):
        print(f"Calibration folder {calibration_folder} does not exist.")
        return

    print(f"Processing calibration folder: {calibration_folder}")

    # Iterate over all files in the calibration folder
    for master_flat in os.listdir(calibration_folder):
        master_flat_path = os.path.join(calibration_folder, master_flat)
        
        # Log the current file being processed
        print(f"Checking file: {master_flat_path}")
        
        if master_flat.startswith('master_flat') and (master_flat.endswith('.fit') or master_flat.endswith('.fits')):
            try:
                # Open the FITS file and extract data
                with fits.open(master_flat_path) as hdul:
                    flat_data = hdul[0].data  # Assuming the flat frame data is in the primary HDU

                # Ensure the data is correctly extracted
                if flat_data is None:
                    print(f"No data found in {master_flat}")
                    continue

                print(f"Processing {master_flat}, data shape: {flat_data.shape}")
                
                # Plot the histogram on a linear scale for pixel values
                plt.figure()
                plt.hist(flat_data.flatten(), bins=100, color='blue', alpha=0.7)
                plt.yscale('linear')  # Set y-axis to linear scale
                plt.title(f'Histogram of Master Flat: {master_flat}')
                plt.xlabel('Pixel Value')
                plt.ylabel('Count')
                
                # Save the histogram
                hist_output_path = os.path.join(output_folder, f'{master_flat}_histogram.png')
                plt.savefig(hist_output_path)
                plt.close()

                print(f"Histogram saved for {master_flat} at {hist_output_path}")
            except Exception as e:
                print(f"Failed to process {master_flat}: {e}")

    print(f"All histograms saved in {output_folder}.")
