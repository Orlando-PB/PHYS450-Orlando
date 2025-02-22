#!/usr/bin/env python3
"""
plots.py

Defines a FITSViewer class that embeds a dark‐themed figure with two axes:
– the upper axis displays the FITS image,
– the lower axis shows a smaller histogram.
Interactive events update external Tkinter label widgets for cursor and
selected source stats. Drag and scroll events enable pan/zoom.
"""

import os, json, numpy as np
from math import cos, radians

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord

# Dark theme colors for the figure
FIG_BG_COLOR = "#2D2D2D"
AXES_BG_COLOR = "#2D2D2D"
TEXT_COLOR   = "white"
# Use a pale blue for the histogram
HIST_COLOR   = "#a8d8ea"

class FITSViewer:
    def __init__(self, fits_file, parent_frame, stats_labels):
        """
        fits_file    : path to the FITS file.
        parent_frame : Tkinter frame where the figure will be embedded.
        stats_labels : dict with keys "image", "cursor", "selected" whose .config()
                       will be updated with stats text.
        """
        self.fits_file = fits_file
        self.stats_labels = stats_labels
        
        # Load FITS data
        with fits.open(fits_file) as hdul:
            self.data = hdul[0].data
            self.header = hdul[0].header
        self.ny, self.nx = self.data.shape
        self.min_val = np.min(self.data)
        self.max_val = np.max(self.data)
        
        # Attempt to load astrometry JSON for WCS and source extraction
        base, _ = os.path.splitext(fits_file)
        json_file = base + "_astrometry_solution.json"
        self.detected_sources = []
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
                self.wcs_object = WCS(naxis=2)
                self.wcs_object.wcs.crpix = [self.nx / 2.0, self.ny / 2.0]
                self.wcs_object.wcs.crval = [ra_center, dec_center]
                theta = np.deg2rad(orientation)
                cd11 = -pixscale_deg * np.cos(theta)
                cd12 =  pixscale_deg * np.sin(theta)
                cd21 =  pixscale_deg * np.sin(theta)
                cd22 =  pixscale_deg * np.cos(theta)
                self.wcs_object.wcs.cd = np.array([[cd11, cd12],
                                                   [cd21, cd22]])
                self.wcs_object.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            else:
                self.wcs_object = WCS(self.header)
            if "sources" in astrometry_data:
                self.detected_sources = astrometry_data["sources"]
        else:
            self.wcs_object = WCS(self.header)
        
        # Fallback: try loading star positions from a text file
        if not self.detected_sources:
            star_file = fits_file.replace('.fits', '_star_positions.txt').replace('.fit', '_star_positions.txt')
            if os.path.exists(star_file):
                with open(star_file, 'r') as f:
                    for line in f:
                        try:
                            x, y = line.strip().split(',')
                            self.detected_sources.append({"xcentroid": float(x), "ycentroid": float(y)})
                        except ValueError:
                            continue
        # Convert source pixel coordinates to world coordinates
        for src in self.detected_sources:
            world = self.wcs_object.wcs_pix2world([[src["xcentroid"], src["ycentroid"]]], 0)[0]
            src["ra"] = float(world[0])
            src["dec"] = float(world[1])
        
        # Update image stats externally (convert total pixels to MP)
        total_pixels = self.data.size / 1e6
        image_stats_text = (
            f"Image Stats:\n"
            f"Total Pixels: {total_pixels:.2f} MP\n"
            f"Mean: {np.mean(self.data):.2f}\n"
            f"Median: {np.median(self.data):.2f}\n"
            f"Std Dev: {np.std(self.data):.2f}\n"
            f"Min: {self.min_val:.2f}\n"
            f"Max: {self.max_val:.2f}"
        )
        self.stats_labels["image"].config(text=image_stats_text)
        self.stats_labels["cursor"].config(text="Cursor Stats:\n")
        self.stats_labels["selected"].config(text="Selected Source Stats:\nNo source selected.\nClick a source to see details.")
        
        # Create the figure with two axes:
        # - Image axis (upper): occupies 65% of vertical space
        # - Histogram axis (lower): occupies 15% of vertical space
        self.fig = Figure(figsize=(6,6), dpi=100)
        self.fig.patch.set_facecolor(FIG_BG_COLOR)
        self.ax_image = self.fig.add_axes([0.05, 0.30, 0.9, 0.65], projection=self.wcs_object)
        self.ax_image.set_facecolor(AXES_BG_COLOR)
        self.ax_image.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
        self.ax_image.set_xlabel("RA (hh:mm:ss)", color=TEXT_COLOR)
        self.ax_image.set_ylabel("DEC (deg)", color=TEXT_COLOR)
        self.im = self.ax_image.imshow(self.data, cmap="gray", origin="lower", vmin=self.min_val, vmax=self.max_val)
        
        self.ax_hist = self.fig.add_axes([0.05, 0.10, 0.9, 0.15])
        self.ax_hist.set_facecolor(AXES_BG_COLOR)
        self.ax_hist.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
        self.ax_hist.hist(self.data.flatten(), bins=1000, color=HIST_COLOR, alpha=0.5)
        self.ax_hist.set_yscale("linear")
        
        # Add vertical hover line to histogram (initially hidden)
        self.hover_line = self.ax_hist.axvline(x=0, color='red', linestyle='--', visible=False)
        
        # Save original zoom limits
        self.original_xlim = self.ax_image.get_xlim()
        self.original_ylim = self.ax_image.get_ylim()
        
        # Create the canvas and embed in the parent frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()
        
        # Initialize state variables for toggles, dragging, and source overlays
        self.auto_stretch = False
        self.log_hist = False
        self.show_sources = False
        self.selected_source_artist = None
        self.source_artists = []
        self._dragging = False
        self._last_event = None
        
        # Connect interactive events
        self.canvas.mpl_connect("motion_notify_event", self.update_cursor_stats)
        self.canvas.mpl_connect("button_press_event", self.on_click)  # for source selection
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
    
    def update_cursor_stats(self, event):
        if event.inaxes == self.ax_image and event.xdata is not None and event.ydata is not None:
            world = self.wcs_object.wcs_pix2world([[event.xdata, event.ydata]], 0)[0]
            ra, dec = world[0], world[1]
            ix, iy = int(round(event.xdata)), int(round(event.ydata))
            pixel_val = self.data[iy, ix] if (0 <= ix < self.nx and 0 <= iy < self.ny) else "N/A"
            cursor_text = f"Cursor Stats:\nRA: {ra:.5f}\nDEC: {dec:.5f}\nPixel Value: {pixel_val}"
            self.stats_labels["cursor"].config(text=cursor_text)
            self.hover_line.set_xdata(pixel_val)
            self.hover_line.set_visible(True)
            self.canvas.draw_idle()
        else:
            self.stats_labels["cursor"].config(text="Cursor Stats:\n")
            self.hover_line.set_visible(False)
            self.canvas.draw_idle()
    
    def on_click(self, event):
        if event.inaxes != self.ax_image or event.xdata is None or event.ydata is None:
            return
        world = self.wcs_object.wcs_pix2world([[event.xdata, event.ydata]], 0)[0]
        click_ra, click_dec = world[0], world[1]
        found = None
        threshold = 0.001  # in degrees (~3.6 arcsec)
        for src in self.detected_sources:
            dra = (click_ra - src["ra"]) * cos(radians((click_dec + src["dec"]) / 2.0))
            ddec = click_dec - src["dec"]
            if np.hypot(dra, ddec) < threshold:
                found = src
                break
        if found:
            src_coord = SkyCoord(ra=found["ra"]*u.deg, dec=found["dec"]*u.deg)
            src_text = (
                f"Selected Source Stats:\n"
                f"Centroid: ({found['xcentroid']:.2f}, {found['ycentroid']:.2f})\n"
                f"RA: {src_coord.ra.to_string(unit=u.hour, sep=':', pad=True, precision=2)}\n"
                f"DEC: {src_coord.dec.to_string(sep=':', pad=True, alwayssign=True, precision=2)}\n"
                f"Flux: {found.get('flux', found.get('Flux', 0)):.2f}"
            )
            self.stats_labels["selected"].config(text=src_text)
            if self.selected_source_artist is not None:
                self.selected_source_artist.remove()
            xsrc, ysrc = found["xcentroid"], found["ycentroid"]
            self.selected_source_artist, = self.ax_image.plot(
                xsrc, ysrc, marker='o', markersize=15,
                markeredgecolor='yellow', markerfacecolor='none', lw=2,
                transform=self.ax_image.get_transform('pixel')
            )
            self.canvas.draw_idle()
        else:
            self.stats_labels["selected"].config(
                text="Selected Source Stats:\nNo source selected.\nClick a source to see details."
            )
            if self.selected_source_artist is not None:
                self.selected_source_artist.remove()
                self.selected_source_artist = None
            self.canvas.draw_idle()
    
    def on_press(self, event):
        if event.inaxes == self.ax_image and event.button == 1:
            self._dragging = True
            self._last_event = event
    
    def on_drag(self, event):
        if self._dragging and event.inaxes == self.ax_image and self._last_event is not None:
            dx = event.x - self._last_event.x
            dy = event.y - self._last_event.y
            inv = self.ax_image.transData.inverted()
            p0 = inv.transform((self._last_event.x, self._last_event.y))
            p1 = inv.transform((event.x, event.y))
            ddata = p0 - p1
            cur_xlim = self.ax_image.get_xlim()
            cur_ylim = self.ax_image.get_ylim()
            self.ax_image.set_xlim(cur_xlim + ddata[0])
            self.ax_image.set_ylim(cur_ylim + ddata[1])
            self._last_event = event
            self.canvas.draw_idle()
    
    def on_release(self, event):
        self._dragging = False
        self._last_event = None
    
    def on_scroll(self, event):
        if event.inaxes != self.ax_image or event.xdata is None or event.ydata is None:
            return
        scale_factor = 0.9 if event.button == "up" else 1.1
        cur_xlim = self.ax_image.get_xlim()
        cur_ylim = self.ax_image.get_ylim()
        xdata = event.xdata
        ydata = event.ydata
        new_width = (cur_xlim[1]-cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1]-cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1]-xdata)/(cur_xlim[1]-cur_xlim[0])
        rely = (cur_ylim[1]-ydata)/(cur_ylim[1]-cur_ylim[0])
        self.ax_image.set_xlim([xdata-new_width*(1-relx), xdata+new_width*relx])
        self.ax_image.set_ylim([ydata-new_height*(1-rely), ydata+new_height*rely])
        self.canvas.draw_idle()
    
    def reset_zoom(self):
        self.ax_image.set_xlim(self.original_xlim)
        self.ax_image.set_ylim(self.original_ylim)
        self.canvas.draw_idle()
    
    def zoom_in(self):
        cur_xlim = self.ax_image.get_xlim()
        cur_ylim = self.ax_image.get_ylim()
        xdata = (cur_xlim[0] + cur_xlim[1]) / 2.0
        ydata = (cur_ylim[0] + cur_ylim[1]) / 2.0
        scale_factor = 0.9
        new_width = (cur_xlim[1]-cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1]-cur_ylim[0]) * scale_factor
        self.ax_image.set_xlim([xdata-new_width/2, xdata+new_width/2])
        self.ax_image.set_ylim([ydata-new_height/2, ydata+new_height/2])
        self.canvas.draw_idle()
    
    def zoom_out(self):
        cur_xlim = self.ax_image.get_xlim()
        cur_ylim = self.ax_image.get_ylim()
        xdata = (cur_xlim[0] + cur_xlim[1]) / 2.0
        ydata = (cur_ylim[0] + cur_ylim[1]) / 2.0
        scale_factor = 1.1
        new_width = (cur_xlim[1]-cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1]-cur_ylim[0]) * scale_factor
        self.ax_image.set_xlim([xdata-new_width/2, xdata+new_width/2])
        self.ax_image.set_ylim([ydata-new_height/2, ydata+new_height/2])
        self.canvas.draw_idle()
    
    def toggle_auto_stretch(self):
        if not self.auto_stretch:
            clim_low = np.percentile(self.data, 1)
            clim_high = np.percentile(self.data, 99.2)
            self.im.set_clim(clim_low, clim_high)
            self.auto_stretch = True
        else:
            self.im.set_clim(self.min_val, self.max_val)
            self.auto_stretch = False
        self.canvas.draw_idle()
    
    def toggle_log(self):
        if not self.log_hist:
            self.ax_hist.set_yscale("log")
            self.log_hist = True
        else:
            self.ax_hist.set_yscale("linear")
            self.log_hist = False
        self.canvas.draw_idle()
    
    def toggle_sources(self):
        if not self.show_sources:
            self.source_artists = []
            for src in self.detected_sources:
                artist, = self.ax_image.plot(
                    src["xcentroid"], src["ycentroid"], marker='o', markersize=10,
                    markeredgecolor='red', markerfacecolor='none', lw=1,
                    transform=self.ax_image.get_transform('pixel')
                )
                self.source_artists.append(artist)
            self.show_sources = True
        else:
            for artist in self.source_artists:
                artist.remove()
            self.source_artists = []
            self.show_sources = False
        self.canvas.draw_idle()
