#!/usr/bin/env python3
"""
plots.py

An interactive FITS viewer that uses astrometry-based WCS calibration and source extraction
results (from a JSON file) to display a FITS image with overlays, a histogram, and an info box.
RA is displayed in hours/min/sec. The info box shows either current position (world coordinates and pixel value)
or, if the mouse is near a detected source, the source info (centroid, RA, DEC, and flux).
Flux is taken from the JSON (using the key "flux" or "Flux").

Usage:
    python plots.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from math import cos, radians

# Global variable for the source scatter artist (for removal when toggling).
source_scatter_wcs = None

def show_fits_info(fits_file):
    """
    Display a FITS image in WCS view with interactive features.
      - Uses an astrometry JSON file (if available) to build a WCS.
      - Loads source extraction results to overlay source markers.
      - Displays image statistics, a histogram, and hover info.
        RA is shown in hours/min/sec.
        The info box displays either current position (RA/DEC and pixel value)
        or, if the mouse is near a detected source, the source info (centroid, RA, DEC, and flux).
    """
    try:
        # Open the FITS file and extract image data and header.
        with fits.open(fits_file) as hdul:
            data = hdul[0].data
            header = hdul[0].header
        ny, nx = data.shape

        # Attempt to load astrometry solution JSON.
        base, ext = os.path.splitext(fits_file)
        json_file = base + "_astrometry_solution.json"
        astrometry_data = None
        detected_sources = []

        # Build the WCS from astrometry JSON if possible.
        wcs_object = None
        if os.path.exists(json_file):
            with open(json_file, "r") as jf:
                astrometry_data = json.load(jf)
            if (astrometry_data.get("Right Ascension") is not None and
                astrometry_data.get("Declination") is not None and
                astrometry_data.get("Pixel Scale (arcsec/pixel)") is not None and
                astrometry_data.get("Orientation (degrees)") is not None):

                ra_center = astrometry_data["Right Ascension"]
                dec_center = astrometry_data["Declination"]
                pixscale_arcsec = astrometry_data["Pixel Scale (arcsec/pixel)"]
                orientation = astrometry_data["Orientation (degrees)"]
                pixscale_deg = pixscale_arcsec / 3600.0

                # Create a new WCS using the calibration.
                wcs_object = WCS(naxis=2)
                wcs_object.wcs.crpix = [nx / 2.0, ny / 2.0]
                wcs_object.wcs.crval = [ra_center, dec_center]
                theta = np.deg2rad(orientation)
                cd11 = -pixscale_deg * np.cos(theta)
                cd12 = pixscale_deg * np.sin(theta)
                cd21 = pixscale_deg * np.sin(theta)
                cd22 = pixscale_deg * np.cos(theta)
                wcs_object.wcs.cd = np.array([[cd11, cd12],
                                              [cd21, cd22]])
                wcs_object.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            else:
                print("Astrometry calibration not found in JSON; using header WCS.")
                wcs_object = WCS(header)

            # Use source extraction results from the JSON if available.
            if "sources" in astrometry_data:
                detected_sources = astrometry_data["sources"]
        else:
            print("Astrometry JSON file not found; using header WCS.")
            wcs_object = WCS(header)

        # Fallback: if no sources are loaded, try a star positions text file.
        if not detected_sources:
            star_positions_file = fits_file.replace('.fits', '_star_positions.txt').replace('.fit', '_star_positions.txt')
            if os.path.exists(star_positions_file):
                with open(star_positions_file, 'r') as f:
                    for line in f:
                        try:
                            x, y = line.strip().split(',')
                            detected_sources.append({"xcentroid": float(x), "ycentroid": float(y)})
                        except ValueError:
                            continue

        # Always recompute the source world coordinates from the pixel centroids.
        for src in detected_sources:
            world = wcs_object.wcs_pix2world([[src["xcentroid"], src["ycentroid"]]], 0)[0]
            src["ra"] = float(world[0])
            src["dec"] = float(world[1])

        # Compute image statistics.
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_dev = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)
        total_pixels = data.size / 1e6  # in megapixels
        lower_auto_percentile = np.percentile(data, 1)
        upper_auto_percentile = np.percentile(data, 99.2)

        # Turn off the default toolbar.
        plt.rcParams['toolbar'] = 'none'
        fig = plt.figure(figsize=(10, 8))

        # Create one axis with a WCS (world coordinates) projection.
        ax_img_wcs = fig.add_axes([0.02, 0.35, 0.65, 0.6], projection=wcs_object)
        img_wcs = ax_img_wcs.imshow(data, cmap='gray', origin='lower',
                                    vmin=min_val, vmax=max_val)
        ax_img_wcs.set_title(f'FITS Image: {os.path.basename(fits_file)}')
        ax_img_wcs.set_xlabel('RA (hh:mm:ss)')
        ax_img_wcs.set_ylabel('DEC (deg)')

        # Create a histogram axis.
        ax_hist = fig.add_axes([0.05, 0.15, 0.65, 0.1])
        n, bins, patches = ax_hist.hist(data.flatten(), bins=1000,
                                        color='blue', alpha=0.5)
        ax_hist.set_yscale('linear')
        ax_hist.set_title('Histogram of Pixel Values')
        ax_hist.set_xlabel('Pixel Value')
        ax_hist.set_ylabel('Count')
        hover_line = ax_hist.axvline(x=0, color='red', linestyle='--', visible=False)

        # Create a stats box for displaying information.
        stats_box = fig.text(0.65, 0.7, '', fontsize=12,
                             bbox=dict(facecolor='white', alpha=0.5))

        # Save the original zoom limits.
        original_xlim = ax_img_wcs.get_xlim()
        original_ylim = ax_img_wcs.get_ylim()

        def update_stats_box(ra_deg, dec_deg, pixel_value=None, source_info=None):
            """
            Update the stats box.
              - ra_deg, dec_deg: world coordinates in degrees.
              - If source_info is provided, show its centroid (pixel),
                RA (formatted in HMS), DEC, and Flux (counts).
              - Otherwise, show the current position (RA/DEC) and pixel value.
            """
            # Convert RA (in degrees) to an HMS string.
            coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
            ra_hms = coord.ra.to_string(unit=u.hour, sep=':', pad=True, precision=2)
            dec_dms = coord.dec.to_string(sep=':', pad=True, alwayssign=True, precision=2)
            stats_text = (f'Total Pixels: {total_pixels:.2f} MP\n'
                          f'Mean: {mean_val:.2f}\n'
                          f'Median: {median_val:.2f}\n'
                          f'Std Dev: {std_dev:.2f}\n'
                          f'Min: {min_val:.2f}\n'
                          f'Max: {max_val:.2f}')
            if source_info is not None:
                # Convert the source's RA to HMS.
                src_coord = SkyCoord(ra=source_info["ra"] * u.deg,
                                     dec=source_info["dec"] * u.deg)
                src_ra_hms = src_coord.ra.to_string(unit=u.hour, sep=':', pad=True, precision=2)
                src_dec_dms = src_coord.dec.to_string(sep=':', pad=True, alwayssign=True, precision=2)
                # Try to get flux from the source info; check both "flux" and "Flux".
                flux = source_info.get("flux")
                if flux is None:
                    flux = source_info.get("Flux", 0)
                stats_text += (f'\n\nSource Info:\n'
                               f'Centroid (pixel): ({source_info["xcentroid"]:.2f}, {source_info["ycentroid"]:.2f})\n'
                               f'RA: {src_ra_hms}\n'
                               f'DEC: {src_dec_dms}\n'
                               f'Flux (counts): {flux:.2f}')
            elif pixel_value is not None:
                stats_text += (f'\n\nCurrent Position:\n'
                               f'RA: {ra_hms}\n'
                               f'DEC: {dec_dms}\n'
                               f'Pixel Value: {pixel_value}')
            stats_box.set_text(stats_text)

        def on_hover(event):
            # Only process events in the WCS image axis.
            if event.inaxes != ax_img_wcs:
                return

            xdata = event.xdata
            ydata = event.ydata
            if xdata is None or ydata is None:
                return

            # Convert the raw pixel coordinates to world coordinates.
            world_coords = wcs_object.wcs_pix2world([[xdata, ydata]], 0)[0]
            mouse_ra, mouse_dec = world_coords[0], world_coords[1]
            print(f"Hover: pixel=({xdata:.2f}, {ydata:.2f}) -> world=({mouse_ra:.5f}, {mouse_dec:.5f})")

            found_source = None
            wcs_threshold_deg = 0.001  # threshold in degrees (~3.6 arcsec)
            for src in detected_sources:
                # Compute distance using a small-angle approximation.
                dra = (mouse_ra - src["ra"]) * cos(radians((mouse_dec + src["dec"]) / 2.0))
                ddec = mouse_dec - src["dec"]
                dist_deg = np.hypot(dra, ddec)
                if dist_deg < wcs_threshold_deg:
                    found_source = src
                    print("Found source near hover:", src)
                    break

            if found_source is not None:
                update_stats_box(found_source["ra"], found_source["dec"],
                                 source_info=found_source)
                hover_line.set_visible(False)
            else:
                # Convert the world coordinates back to pixel coordinates to sample the image data.
                pix_coords = wcs_object.wcs_world2pix([[mouse_ra, mouse_dec]], 0)[0]
                if np.isfinite(pix_coords[0]) and np.isfinite(pix_coords[1]):
                    ix, iy = int(pix_coords[0]), int(pix_coords[1])
                    if 0 <= ix < nx and 0 <= iy < ny:
                        pixel_val = data[iy, ix]
                        update_stats_box(mouse_ra, mouse_dec, pixel_value=pixel_val)
                        hover_line.set_xdata([pixel_val])
                        hover_line.set_visible(True)
                    else:
                        hover_line.set_visible(False)
                else:
                    hover_line.set_visible(False)
            fig.canvas.draw_idle()

        # Connect the hover callback.
        fig.canvas.mpl_connect('motion_notify_event', on_hover)

        # Simple pan/drag support.
        def on_press(event):
            if event.inaxes == ax_img_wcs and event.button == 1:
                on_press.lastx, on_press.lasty = event.xdata, event.ydata
                ax_img_wcs._dragging = True

        def on_release(event):
            ax_img_wcs._dragging = False

        def on_drag(event):
            if getattr(ax_img_wcs, '_dragging', False) and event.inaxes == ax_img_wcs:
                dx = event.xdata - on_press.lastx
                dy = event.ydata - on_press.lasty
                cur_xlim = ax_img_wcs.get_xlim()
                cur_ylim = ax_img_wcs.get_ylim()
                ax_img_wcs.set_xlim([cur_xlim[0] - dx, cur_xlim[1] - dx])
                ax_img_wcs.set_ylim([cur_ylim[0] - dy, cur_ylim[1] - dy])
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('motion_notify_event', on_drag)

        # Zooming with the scroll wheel.
        def on_zoom(event):
            if event.inaxes == ax_img_wcs:
                scale_factor = 1.1 if event.button == 'up' else 0.9
                cur_xlim = ax_img_wcs.get_xlim()
                cur_ylim = ax_img_wcs.get_ylim()
                xdata, ydata = event.xdata, event.ydata
                if xdata is None or ydata is None:
                    return
                new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
                new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
                relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
                rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
                ax_img_wcs.set_xlim([xdata - new_width * (1 - relx),
                                     xdata + new_width * relx])
                ax_img_wcs.set_ylim([ydata - new_height * (1 - rely),
                                     ydata + new_height * rely])
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect('scroll_event', on_zoom)

        # Reset Zoom Button.
        ax_reset = plt.axes([0.75, 0.05, 0.15, 0.05])
        reset_button = Button(ax_reset, 'Reset Zoom')

        def reset_zoom(event):
            ax_img_wcs.set_xlim(original_xlim)
            ax_img_wcs.set_ylim(original_ylim)
            fig.canvas.draw_idle()

        reset_button.on_clicked(reset_zoom)

        # Auto-stretch toggle.
        ax_auto_toggle = plt.axes([0.75, 0.2, 0.15, 0.05])
        auto_toggle_button = Button(ax_auto_toggle, 'Auto Stretch: Off')
        is_auto_stretched = [False]

        def toggle_auto_stretch(event):
            if not is_auto_stretched[0]:
                img_wcs.set_clim(lower_auto_percentile, upper_auto_percentile)
                auto_toggle_button.label.set_text('Auto Stretch: On')
            else:
                img_wcs.set_clim(min_val, max_val)
                auto_toggle_button.label.set_text('Auto Stretch: Off')
            is_auto_stretched[0] = not is_auto_stretched[0]
            fig.canvas.draw_idle()

        auto_toggle_button.on_clicked(toggle_auto_stretch)

        # Log scale histogram toggle.
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

        # Toggle source overlay.
        ax_source_toggle = plt.axes([0.75, 0.25, 0.15, 0.05])
        source_toggle_button = Button(ax_source_toggle, 'Sources: Off')
        are_sources_displayed = [False]

        def toggle_sources(event):
            global source_scatter_wcs
            if not are_sources_displayed[0]:
                if detected_sources:
                    # Convert source pixel positions to world coordinates.
                    pixel_coords = np.array([[src["xcentroid"], src["ycentroid"]] for src in detected_sources])
                    world_coords = wcs_object.wcs_pix2world(pixel_coords, 0)
                    source_scatter_wcs = ax_img_wcs.scatter(
                        world_coords[:, 0], world_coords[:, 1],
                        s=50, edgecolors='red', facecolors='none', lw=1,
                        transform=ax_img_wcs.get_transform('world')
                    )
                source_toggle_button.label.set_text('Sources: On')
            else:
                if source_scatter_wcs is not None:
                    source_scatter_wcs.remove()
                    source_scatter_wcs = None
                source_toggle_button.label.set_text('Sources: Off')
            are_sources_displayed[0] = not are_sources_displayed[0]
            fig.canvas.draw_idle()

        source_toggle_button.on_clicked(toggle_sources)

        plt.show()

    except Exception as e:
        print(f"Error processing {fits_file}: {e}")

if __name__ == '__main__':
    # Replace with the path to your FITS file.
    test_fits = "path/to/your/image.fits"
    show_fits_info(test_fits)
