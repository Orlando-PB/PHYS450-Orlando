# Simple Astronomy (PHYS450)

> **Status:** finished university project (Lancaster University, PHYS450). Archived, not maintained.

A desktop app for reducing and calibrating telescope images, written in Python with a Tkinter interface. Point it at a folder of FITS files and it takes them from raw frames to calibrated, plate-solved images with measured star brightnesses.

## What it does

1. **Sorts** raw FITS files into bias, dark, flat and light frames (`utils.py`).
2. **Calibrates** the light frames with master bias, dark and flat frames (`fits_processor.py`).
3. **Plate-solves** each image through the [nova.astrometry.net](https://nova.astrometry.net) API to get its position, pixel scale and orientation on the sky (`astrometry.py`).
4. **Detects stars** with photutils' `DAOStarFinder` and converts pixel positions to RA/Dec (`source_extraction.py`).
5. **Calibrates the photometry** by matching detected stars to the Gaia catalogue and fitting instrumental against Gaia magnitudes, with sigma clipping (`photometric_calibration.py`). Example fit: `Photometric_Calibration_Fit.png`.
6. **Displays** the images with histogram, stretch controls and detected sources overlaid (`plots.py`).

Images are processed in parallel, with progress shown in the app.

## Running it

```sh
pip install numpy astropy astroquery photutils matplotlib pillow requests psutil
python main.py
```

Plate solving needs a free astrometry.net API key: `export ASTROMETRY_API_KEY=...` before running. `demo/` and `demo small/` hold sample data.

## Related

Lancaster "Astrolab" equipment inventory: https://docs.google.com/spreadsheets/d/1ufNTqUrc_-x576DD8k2ixLflX1zlaSnAk7NTWtC2-KA/edit?usp=sharing
