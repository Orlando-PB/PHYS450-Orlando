import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from astropy.io import fits
import numpy as np
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
import threading

def show_fits_info(fits_file):
    """
    Opens a FITS file, shows its statistics (mean, min, max, median, total pixels, standard deviation, and current pixel value),
    and plots both a histogram of pixel values and the image itself. A toggle button allows switching between X,Y and RA/DEC coordinates,
    and another toggle can circle stars if positions are available. It includes zooming and panning functionality.
    """

    try:
        # Open the FITS file
        with fits.open(fits_file) as hdul:
            data = hdul[0].data  # Assuming the data is in the primary HDU
            header = hdul[0].header  # Get the FITS header for RA/DEC info

            if data is None:
                print(f"No data found in {fits_file}")
                return

            # Check if star positions file exists for this FITS file
            star_positions_file = fits_file.replace('.fits', '_star_positions.txt').replace('.fit', '_star_positions.txt')
            star_positions = None
            if os.path.exists(star_positions_file):
                with open(star_positions_file, 'r') as f:
                    star_positions = []
                    for line in f:
                        try:
                            x, y = line.strip().split(',')
                            star_positions.append((float(x), float(y)))
                        except ValueError:
                            continue  # Skip invalid lines
            else:
                print("No star positions available for this file.")

            # Calculate statistics
            mean_val = np.mean(data)
            min_val = np.min(data)
            max_val = np.max(data)
            median_val = np.median(data)
            std_dev = np.std(data)
            total_pixels = data.size / 1e6  # Convert to megapixels

            # Calculate auto-stretch limits (e.g., percentiles for contrast stretching)
            lower_auto_percentile = np.percentile(data, 1)
            upper_auto_percentile = np.percentile(data, 99.2)

            # Remove the default toolbar
            plt.rcParams['toolbar'] = 'none'

            # Set up the figure and axes with custom layout
            fig = plt.figure(figsize=(10, 8))  # Adjust figure size as needed

            # Image plot (large, taking up most of the top)
            ax_img = fig.add_axes([0.02, 0.35, 0.65, 0.6])  # Adjusted position and size
            img = ax_img.imshow(data, cmap='gray', origin='lower', vmin=min_val, vmax=max_val)
            ax_img.set_title(f'Image of FITS File: {os.path.basename(fits_file)}')
            ax_img.set_xlabel('X Pixel')
            ax_img.set_ylabel('Y Pixel')

            # Hover line for histogram
            hover_line = None

            # Histogram (smaller, below the image, 1:5 height ratio)
            ax_hist = fig.add_axes([0.05, 0.15, 0.65, 0.1])  # Adjusted position and size
            n, bins, patches = ax_hist.hist(data.flatten(), bins=1000, color='blue', alpha=0.5)
            ax_hist.set_yscale('linear')  # Start with linear scale
            ax_hist.set_title('Histogram of Pixel Values')
            ax_hist.set_xlabel('Pixel Value')
            ax_hist.set_ylabel('Count')

            # Add vertical line for current pixel value (red bar on histogram)
            hover_line = ax_hist.axvline(x=0, color='red', linestyle='--', visible=False)

            # Stats box (to the right of the image, including current pixel info)
            stats_box = fig.text(0.65, 0.7, '', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

            # Variables to track dragging state for panning
            dragging = [False]
            is_ra_dec = [False]
            zoom_level = [1.0]

            # Thread for RA/DEC calculation
            hover_thread = [None]

            # Hover functionality, including current pixel info (X,Y or RA/DEC)
            def update_stats_box(x, y, pixel_value=None):
                stats_text = (f'Total Pixels: {total_pixels:.2f} MP\n'
                              f'Mean: {mean_val:.2f}\n'
                              f'Median: {median_val:.2f}\n'
                              f'Standard Deviation: {std_dev:.2f}\n'
                              f'Min: {min_val:.2f}\n'
                              f'Max: {max_val:.2f}')
                
                if pixel_value is not None:
                    if is_ra_dec[0]:
                        # Calculate RA/DEC from pixel coordinates using WCS info in header
                        w = WCS(header)
                        ra_dec = w.pixel_to_world(x, y)
                        ra = ra_dec.ra.degree  # RA in degrees
                        dec = ra_dec.dec.degree  # DEC in degrees

                        # Convert RA from degrees to hours
                        skycoord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
                        ra_hours = skycoord.ra.hms

                        # Update the stats text with RA in both degrees and hours, and DEC in degrees
                        stats_text += (f'\n\nCurrent Pixel (RA, DEC):\n'
                                       f'RA: {ra:.6f}° ({int(ra_hours.h):02d}h {int(ra_hours.m):02d}m {ra_hours.s:.2f}s)\n'
                                       f'DEC: {dec:.6f}°\n'
                                       f'Value: {pixel_value}')
                    else:
                        stats_text += f'\n\nCurrent Pixel: ({x}, {y})\nValue: {pixel_value}'

                stats_box.set_text(stats_text)

            def on_hover(event):
                if not dragging[0] and event.inaxes == ax_img:
                    x, y = int(event.xdata), int(event.ydata)

                    if 0 <= x < data.shape[1] and 0 <= y < data.shape[0]:
                        pixel_value = data[y, x]

                        # Kill previous thread if running
                        if hover_thread[0] is not None and hover_thread[0].is_alive():
                            hover_thread[0].join()

                        # Start a new thread for RA/DEC calculation
                        hover_thread[0] = threading.Thread(target=update_stats_box, args=(x, y, pixel_value))
                        hover_thread[0].start()

                        # Update the red bar position on the histogram
                        hover_line.set_xdata([pixel_value])
                        hover_line.set_visible(True)

                        fig.canvas.draw_idle()

            # Connect the hover event
            fig.canvas.mpl_connect('motion_notify_event', on_hover)

            # Panning functionality
            def on_press(event):
                if event.inaxes == ax_img and event.button == 1:  # Left mouse button for dragging
                    dragging[0] = True
                    on_press.lastx, on_press.lasty = event.xdata, event.ydata

            def on_release(event):
                dragging[0] = False  # Stop dragging

            def on_drag(event):
                if dragging[0] and event.inaxes == ax_img:
                    dx = event.xdata - on_press.lastx
                    dy = event.ydata - on_press.lasty
                    cur_xlim = ax_img.get_xlim()
                    cur_ylim = ax_img.get_ylim()
                    ax_img.set_xlim([cur_xlim[0] - dx, cur_xlim[1] - dx])
                    ax_img.set_ylim([cur_ylim[0] - dy, cur_ylim[1] - dy])
                    fig.canvas.draw_idle()

            # Zoom functionality
            def on_zoom(event):
                if event.inaxes == ax_img:
                    scale_factor = 1.1 if event.button == 'up' else 0.9

                    cur_xlim = ax_img.get_xlim()
                    cur_ylim = ax_img.get_ylim()

                    xdata = event.xdata
                    ydata = event.ydata

                    new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
                    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

                    relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
                    rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

                    ax_img.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * (relx)])
                    ax_img.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * (rely)])

                    fig.canvas.draw_idle()

            # Reset zoom button
            ax_reset = plt.axes([0.75, 0.05, 0.15, 0.05])  # Adjusted position
            reset_button = Button(ax_reset, 'Reset Zoom')

            original_xlim = ax_img.get_xlim()
            original_ylim = ax_img.get_ylim()

            def reset_zoom(event):
                ax_img.set_xlim(original_xlim)
                ax_img.set_ylim(original_ylim)
                fig.canvas.draw_idle()

            reset_button.on_clicked(reset_zoom)

            # Connect dragging and zooming events
            fig.canvas.mpl_connect('button_press_event', on_press)
            fig.canvas.mpl_connect('button_release_event', on_release)
            fig.canvas.mpl_connect('motion_notify_event', on_drag)
            fig.canvas.mpl_connect('scroll_event', on_zoom)

            # Toggle display of circles around stars (if positions are available)
            if star_positions:
                circles = []
                are_stars_displayed = [False]
                ax_star_toggle = plt.axes([0.75, 0.25, 0.15, 0.05])
                star_toggle_button = Button(ax_star_toggle, 'Stars: Off')

                def toggle_stars(event):
                    if not are_stars_displayed[0]:
                        # Display circles around stars
                        for (x, y) in star_positions:
                            circle = plt.Circle((x, y), 10, color='red', fill=False, lw=1)
                            ax_img.add_artist(circle)
                            circles.append(circle)
                        star_toggle_button.label.set_text('Stars: On')
                    else:
                        # Remove circles
                        for circle in circles:
                            circle.remove()
                        circles.clear()
                        star_toggle_button.label.set_text('Stars: Off')

                    are_stars_displayed[0] = not are_stars_displayed[0]
                    fig.canvas.draw_idle()

                star_toggle_button.on_clicked(toggle_stars)
            else:
                print("No star positions available for this file.")

            # Add RA/DEC toggle button
            ax_coord_toggle = plt.axes([0.75, 0.3, 0.15, 0.05])  # Adjusted position
            coord_toggle_button = Button(ax_coord_toggle, 'Coord: X,Y')

            def toggle_coords(event):
                if not is_ra_dec[0]:
                    w = WCS(header)
                    x_pix, y_pix = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
                    ra_dec = w.pixel_to_world(x_pix, y_pix)
                    
                    ra_deg = ra_dec.ra.degree
                    dec_deg = ra_dec.dec.degree
                    
                    ax_img.set_xlabel('RA (deg)')
                    ax_img.set_ylabel('DEC (deg)')
                    coord_toggle_button.label.set_text('Coord: RA/DEC')
                else:
                    ax_img.set_xlabel('X Pixel')
                    ax_img.set_ylabel('Y Pixel')
                    coord_toggle_button.label.set_text('Coord: X,Y')

                is_ra_dec[0] = not is_ra_dec[0]
                fig.canvas.draw_idle()

            coord_toggle_button.on_clicked(toggle_coords)

            # Auto-Stretch toggle button
            ax_toggle = plt.axes([0.75, 0.2, 0.15, 0.05])  # Adjusted position
            toggle_button = Button(ax_toggle, 'Auto Stretch: Off')

            is_auto_stretched = [False]

            def toggle_display(event):
                if not is_auto_stretched[0]:
                    img.set_clim(lower_auto_percentile, upper_auto_percentile)
                    toggle_button.label.set_text('Auto Stretch: On')
                else:
                    img.set_clim(min_val, max_val)
                    toggle_button.label.set_text('Auto Stretch: Off')

                is_auto_stretched[0] = not is_auto_stretched[0]
                fig.canvas.draw_idle()

            toggle_button.on_clicked(toggle_display)

            # Log histogram toggle button
            ax_log_toggle = plt.axes([0.75, 0.1, 0.15, 0.05])  # Adjusted position
            log_toggle_button = Button(ax_log_toggle, 'Log Hist: Off')

            is_log = [False]

            def toggle_log(event):
                if not is_log[0]:
                    ax_hist.set_yscale('log')
                    log_toggle_button.label.set_text('Log Hist: On')
                else:
                    ax_hist.set_yscale('linear')
                    log_toggle_button.label.set_text('Log Hist: Off')

                is_log[0] = not is_log[0]
                fig.canvas.draw_idle()

            log_toggle_button.on_clicked(toggle_log)

            # Show the plot
            plt.show()

    except Exception as e:
        print(f"Error processing {fits_file}: {e}")

