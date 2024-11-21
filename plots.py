import os
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import WCSAxes  # Import WCSAxes for WCS projection

def show_fits_info(fits_file):
    # Function to View a FITS file

    try:
        # Open the FITS file
        with fits.open(fits_file) as hdul:
            data = hdul[0].data
            header = hdul[0].header  # Get the FITS header

            # Check if star positions file is generated for this FITS file - Needs improvement
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
            total_pixels = data.size / 1e6  # Converting to megapixels

            # Calculate auto-stretch
            lower_auto_percentile = np.percentile(data, 1)
            upper_auto_percentile = np.percentile(data, 99.2)

            # Remove the default toolbar
            plt.rcParams['toolbar'] = 'none'

            # Set up the figure
            fig = plt.figure(figsize=(10, 8))

            # World Coordinate System (WCS)
            w = WCS(header)

            # Variables to track coordinate system
            is_ra_dec = [False]

            # Create axes for pixel coordinates
            ax_img_pix = fig.add_axes([0.02, 0.35, 0.65, 0.6])
            img_pix = ax_img_pix.imshow(data, cmap='gray', origin='lower', vmin=min_val, vmax=max_val)
            ax_img_pix.set_title(f'Image of FITS File: {os.path.basename(fits_file)}')
            ax_img_pix.set_xlabel('X Pixel')
            ax_img_pix.set_ylabel('Y Pixel')

            # Create axes for WCS (RA/DEC) coordinates
            ax_img_wcs = fig.add_axes([0.02, 0.35, 0.65, 0.6], projection=w)
            img_wcs = ax_img_wcs.imshow(data, cmap='gray', origin='lower', vmin=min_val, vmax=max_val)
            ax_img_wcs.set_title(f'Image of FITS File: {os.path.basename(fits_file)}')
            ax_img_wcs.set_xlabel('RA (deg)')
            ax_img_wcs.set_ylabel('DEC (deg)')
            ax_img_wcs.set_visible(False)  # Hide WCS axes initially

            # Initialize current axes and image
            current_ax = [ax_img_pix]
            current_img = [img_pix]

            # Hover line for histogram
            hover_line = None

            # Histogram
            ax_hist = fig.add_axes([0.05, 0.15, 0.65, 0.1])
            n, bins, patches = ax_hist.hist(data.flatten(), bins=1000, color='blue', alpha=0.5)
            ax_hist.set_yscale('linear')
            ax_hist.set_title('Histogram of Pixel Values')
            ax_hist.set_xlabel('Pixel Value')
            ax_hist.set_ylabel('Count')

            # Current pixel value
            hover_line = ax_hist.axvline(x=0, color='red', linestyle='--', visible=False)

            # info box
            stats_box = fig.text(0.65, 0.7, '', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

            # Variables to track dragging state for panning
            dragging = [False]
            zoom_level = [1.0]

            # Thread for RA/DEC calculation
            hover_thread = [None]

            # Update statistics box
            def update_stats_box(x, y, pixel_value=None):
                stats_text = (f'Total Pixels: {total_pixels:.2f} MP\n'
                              f'Mean: {mean_val:.2f}\n'
                              f'Median: {median_val:.2f}\n'
                              f'Standard Deviation: {std_dev:.2f}\n'
                              f'Min: {min_val:.2f}\n'
                              f'Max: {max_val:.2f}')
                
                if pixel_value is not None:
                    if is_ra_dec[0]:
                        # Already in RA/DEC
                        ra = x
                        dec = y

                        # Convert RA from degrees to hours
                        skycoord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
                        ra_hours = skycoord.ra.hms

                        # Update the stats text with RA in both degrees and hours, and DEC in degrees
                        stats_text += (f'\n\nCurrent Position (RA, DEC):\n'
                                       f'RA: {ra:.6f}° ({int(ra_hours.h):02d}h {int(ra_hours.m):02d}m {ra_hours.s:.2f}s)\n'
                                       f'DEC: {dec:.6f}°\n'
                                       f'Value: {pixel_value}')
                    else:
                        stats_text += f'\n\nCurrent Pixel: ({x}, {y})\nValue: {pixel_value}'

                stats_box.set_text(stats_text)

            # Hover Functionality
            def on_hover(event):
                if not dragging[0] and event.inaxes == current_ax[0]:
                    xdata = event.xdata
                    ydata = event.ydata

                    if xdata is None or ydata is None:
                        return  # Ignore if outside the plot area

                    if is_ra_dec[0]:
                        # In WCS coordinates
                        sky_coord = SkyCoord(xdata * u.deg, ydata * u.deg, frame='icrs')
                        x, y = w.world_to_pixel(sky_coord)
                        x = int(x)
                        y = int(y)
                        ra = xdata
                        dec = ydata
                    else:
                        x = int(xdata)
                        y = int(ydata)

                    if 0 <= x < data.shape[1] and 0 <= y < data.shape[0]:
                        pixel_value = data[y, x]

                        # Kill previous thread if running
                        if hover_thread[0] is not None and hover_thread[0].is_alive():
                            hover_thread[0].join()

                        # Start a new thread for stats box update
                        if is_ra_dec[0]:
                            hover_thread[0] = threading.Thread(target=update_stats_box, args=(ra, dec, pixel_value))
                        else:
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
                if event.inaxes == current_ax[0] and event.button == 1:  # Left mouse button for dragging
                    dragging[0] = True
                    on_press.lastx, on_press.lasty = event.xdata, event.ydata

            def on_release(event):
                dragging[0] = False  # Stop dragging

            def on_drag(event):
                if dragging[0] and event.inaxes == current_ax[0]:
                    dx = event.xdata - on_press.lastx
                    dy = event.ydata - on_press.lasty
                    cur_xlim = current_ax[0].get_xlim()
                    cur_ylim = current_ax[0].get_ylim()
                    current_ax[0].set_xlim([cur_xlim[0] - dx, cur_xlim[1] - dx])
                    current_ax[0].set_ylim([cur_ylim[0] - dy, cur_ylim[1] - dy])
                    fig.canvas.draw_idle()

            # Zoom functionality
            def on_zoom(event):
                if event.inaxes == current_ax[0]:
                    scale_factor = 1.1 if event.button == 'up' else 0.9

                    cur_xlim = current_ax[0].get_xlim()
                    cur_ylim = current_ax[0].get_ylim()

                    xdata = event.xdata
                    ydata = event.ydata

                    new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
                    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

                    relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
                    rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

                    current_ax[0].set_xlim([xdata - new_width * (1 - relx), xdata + new_width * (relx)])
                    current_ax[0].set_ylim([ydata - new_height * (1 - rely), ydata + new_height * (rely)])

                    fig.canvas.draw_idle()

            # Reset zoom button
            ax_reset = plt.axes([0.75, 0.05, 0.15, 0.05])  # Adjusted position
            reset_button = Button(ax_reset, 'Reset Zoom')

            # Store original limits for both axes
            original_xlim_pix = ax_img_pix.get_xlim()
            original_ylim_pix = ax_img_pix.get_ylim()
            original_xlim_wcs = ax_img_wcs.get_xlim()
            original_ylim_wcs = ax_img_wcs.get_ylim()

            def reset_zoom(event):
                if is_ra_dec[0]:
                    current_ax[0].set_xlim(original_xlim_wcs)
                    current_ax[0].set_ylim(original_ylim_wcs)
                else:
                    current_ax[0].set_xlim(original_xlim_pix)
                    current_ax[0].set_ylim(original_ylim_pix)
                fig.canvas.draw_idle()

            reset_button.on_clicked(reset_zoom)

            # Connect dragging and zooming events
            fig.canvas.mpl_connect('button_press_event', on_press)
            fig.canvas.mpl_connect('button_release_event', on_release)
            fig.canvas.mpl_connect('motion_notify_event', on_drag)
            fig.canvas.mpl_connect('scroll_event', on_zoom)

            # Toggle display of circles around stars
            if star_positions:
                circles_pix = []
                circles_wcs = []
                are_stars_displayed = [False]
                ax_star_toggle = plt.axes([0.75, 0.25, 0.15, 0.05])
                star_toggle_button = Button(ax_star_toggle, 'Stars: Off')

                def toggle_stars(event):
                    if not are_stars_displayed[0]:
                        # Display circles around stars
                        for (x, y) in star_positions:
                            # For pixel axes
                            circle_pix = plt.Circle((x, y), 10, color='red', fill=False, lw=1)
                            ax_img_pix.add_artist(circle_pix)
                            circles_pix.append(circle_pix)

                            # For WCS axes
                            if is_ra_dec[0]:
                                sky_coord = w.pixel_to_world(x, y)
                                circle_wcs = plt.Circle((sky_coord.ra.deg, sky_coord.dec.deg), 0.01, transform=ax_img_wcs.get_transform('fk5'), color='red', fill=False, lw=1)
                                ax_img_wcs.add_artist(circle_wcs)
                                circles_wcs.append(circle_wcs)

                        star_toggle_button.label.set_text('Stars: On')
                    else:
                        # Remove circles
                        for circle in circles_pix:
                            circle.remove()
                        circles_pix.clear()
                        for circle in circles_wcs:
                            circle.remove()
                        circles_wcs.clear()
                        star_toggle_button.label.set_text('Stars: Off')

                    are_stars_displayed[0] = not are_stars_displayed[0]
                    fig.canvas.draw_idle()

                star_toggle_button.on_clicked(toggle_stars)
            else:
                print("No star positions available for this file.")

            # Add RA/DEC toggle button
            ax_coord_toggle = plt.axes([0.75, 0.3, 0.15, 0.05])
            coord_toggle_button = Button(ax_coord_toggle, 'Coord: X,Y')

            def toggle_coords(event):
                if not is_ra_dec[0]:
                    # Switch to RA/DEC axes
                    ax_img_pix.set_visible(False)
                    ax_img_wcs.set_visible(True)
                    current_ax[0] = ax_img_wcs
                    current_img[0] = img_wcs
                    coord_toggle_button.label.set_text('Coord: RA/DEC')
                else:
                    # Switch to pixel axes
                    ax_img_pix.set_visible(True)
                    ax_img_wcs.set_visible(False)
                    current_ax[0] = ax_img_pix
                    current_img[0] = img_pix
                    coord_toggle_button.label.set_text('Coord: X,Y')
                is_ra_dec[0] = not is_ra_dec[0]
                fig.canvas.draw_idle()

            coord_toggle_button.on_clicked(toggle_coords)

            # Auto-Stretch toggle button
            ax_toggle = plt.axes([0.75, 0.2, 0.15, 0.05])
            toggle_button = Button(ax_toggle, 'Auto Stretch: Off')

            is_auto_stretched = [False]

            def toggle_display(event):
                if not is_auto_stretched[0]:
                    current_img[0].set_clim(lower_auto_percentile, upper_auto_percentile)
                    toggle_button.label.set_text('Auto Stretch: On')
                else:
                    current_img[0].set_clim(min_val, max_val)
                    toggle_button.label.set_text('Auto Stretch: Off')

                is_auto_stretched[0] = not is_auto_stretched[0]
                fig.canvas.draw_idle()

            toggle_button.on_clicked(toggle_display)

            # Log histogram toggle button
            ax_log_toggle = plt.axes([0.75, 0.1, 0.15, 0.05])
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
