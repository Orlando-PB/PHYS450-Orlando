# fits_processor.py

import os
import json
import numpy as np
from astropy.io import fits
import astrometry
from utils import sort_files_into_subfolders
from source_extraction import extract_sources
import concurrent.futures
import threading
from photometric_calibration import perform_photometric_calibration


ASTROMETRY_SEMAPHORE = threading.Semaphore(4)
PHOTOMETRY_SEMAPHORE = threading.Semaphore(1)

def process_light_images(base_folder, output_folder,
                         use_flats=True, use_darks=True, use_biases=True,
                         do_astrometry=True, astrometry_api_key=None,
                         progress_callback=None,
                         max_workers=3):


    def _report(msg, **kwargs):
        if progress_callback:
            progress_callback(msg, **kwargs)
        else:
            print(msg)

    sorted_categories = sort_files_into_subfolders(base_folder)

    bias_count = len(sorted_categories['Bias']) if use_biases else 0
    dark_count = len(sorted_categories['Dark']) if use_darks else 0
    flat_count = sum(len(flats) for flats in sorted_categories['Flat'].values()) if use_flats else 0
    light_count = sum(len(files) for files in sorted_categories['Light'].values())

    calibration_folder = os.path.join(output_folder, 'calibration')
    os.makedirs(calibration_folder, exist_ok=True)

    master_dark = master_bias = None

    # -------------------------------------------------------------------
    # 1) Create Master Bias
    # -------------------------------------------------------------------
    if use_biases and bias_count > 0:
        _report(f"Bias: 0/{bias_count}", category="bias", current=0, total=bias_count, done=False)
        try:
            create_master_bias(os.path.join(base_folder, 'Bias'), calibration_folder,
                               progress_callback=lambda i: _report(f"Bias: {i}/{bias_count}",
                                                                   category="bias", current=i, total=bias_count))
        except Exception as e:
            _report(f"Error creating master bias: {e}", color="red")
        else:
            master_bias = load_master_frame(calibration_folder, "master_bias")
            _report(f"Bias: {bias_count}/{bias_count} (done)", category="bias",
                    current=bias_count, total=bias_count, done=True, color="green")
    else:
        _report(f"Bias: 0/0 (done)", category="bias", current=0, total=0, done=True, color="green")

    # -------------------------------------------------------------------
    # 2) Create Master Dark
    # -------------------------------------------------------------------
    if use_darks and dark_count > 0:
        _report(f"Darks: 0/{dark_count}", category="dark", current=0, total=dark_count, done=False)
        try:
            create_master_dark(os.path.join(base_folder, 'Darks'), calibration_folder, master_bias,
                               progress_callback=lambda i: _report(f"Darks: {i}/{dark_count}",
                                                                   category="dark", current=i, total=dark_count))
        except Exception as e:
            _report(f"Error creating master dark: {e}", color="red")
        else:
            master_dark = load_master_frame(calibration_folder, "master_dark")
            _report(f"Darks: {dark_count}/{dark_count} (done)", category="dark",
                    current=dark_count, total=dark_count, done=True, color="green")
    else:
        _report(f"Darks: 0/0 (done)", category="dark", current=0, total=0, done=True, color="green")

    # -------------------------------------------------------------------
    # 3) Create Master Flats
    # -------------------------------------------------------------------
    if use_flats and flat_count > 0:
        _report(f"Flats: 0/{flat_count}", category="flat", current=0, total=flat_count, done=False)
        try:
            create_master_flats_for_filters(os.path.join(base_folder, 'Flats'), calibration_folder, master_bias,
                                            progress_callback=lambda c: _report(f"Flats: {c}/{flat_count}",
                                                                                category="flat", current=c, total=flat_count))
        except Exception as e:
            _report(f"Error creating master flats: {e}", color="red")
        else:
            _report(f"Flats: {flat_count}/{flat_count} (done)", category="flat",
                    current=flat_count, total=flat_count, done=True, color="green")
    else:
        _report(f"Flats: 0/0 (done)", category="flat", current=0, total=0, done=True, color="green")

    # -------------------------------------------------------------------
    # 4) Setup astrometry
    # -------------------------------------------------------------------
    astrometry_session = None
    if do_astrometry:
        try:
            if astrometry_api_key:
                astrometry.API_KEY = astrometry_api_key
            astrometry_session = astrometry.setup_astrometry()
        except Exception as e:
            _report(f"Error setting up astrometry: {e}", color="red")
            astrometry_session = None

    # -------------------------------------------------------------------
    # 5) Prepare to process Light frames
    # -------------------------------------------------------------------
    light_folder = os.path.join(base_folder, 'Lights')
    calibrated_folder = os.path.join(output_folder, 'calibrated')
    os.makedirs(calibrated_folder, exist_ok=True)

    processed_lights = 0

    gaia_filter_map = {
        "B": "phot_bp_mean_mag",
        "R": "phot_rp_mean_mag",
        "G": "phot_g_mean_mag",
        "L": "phot_g_mean_mag",
    }

    def calibrate_and_annotate(light_path, filter_name):

        nonlocal master_bias, master_dark
        _report(f"Processing light: {os.path.basename(light_path)}")

        # -------------------
        # (A) Calibration
        # -------------------
        with fits.open(light_path) as hdul:
            light_data = hdul[0].data.astype(np.float32)
            header = hdul[0].header

        if master_bias is not None:
            light_data -= master_bias
            light_data = np.clip(light_data, 0, None)

        if master_dark is not None:
            light_data -= master_dark
            light_data = np.clip(light_data, 0, None)

        mf_path = os.path.join(calibration_folder, f"master_flat_{filter_name}.fit")
        if os.path.exists(mf_path):
            mf = load_master_frame(calibration_folder, f"master_flat_{filter_name}")
            mf[mf == 0] = 1.0
            light_data /= mf

        # Save calibrated image
        out_dir = os.path.join(calibrated_folder, filter_name)
        os.makedirs(out_dir, exist_ok=True)
        out_name = os.path.join(out_dir, f"calibrated_{os.path.basename(light_path)}")
        fits.writeto(out_name, light_data, header, overwrite=True)
        _report(f"Calibrated frame saved: {os.path.basename(out_name)}")

        # -------------------
        # (B) Astrometry
        # -------------------
        if do_astrometry and astrometry_session:
            with ASTROMETRY_SEMAPHORE: 
                try:
                    _report(f"Running astrometry on {os.path.basename(out_name)}")
                    astrometry_result = astrometry.process_image(out_name, astrometry_session)
                    _report(
                        f"Astrometry success: RA={astrometry_result.get('Right Ascension')}, "
                        f"DEC={astrometry_result.get('Declination')}"
                    )
                except Exception as e:
                    _report(f"Error during astrometry for {out_name}: {e}", color="red")

        # -------------------
        # (C) Source Extraction & Photometric Calibration
        # -------------------
        try:
            _report(f"Source extraction on {os.path.basename(out_name)}")
            sources = extract_sources(out_name)
            base_name = os.path.splitext(os.path.basename(out_name))[0]
            json_filename = os.path.join(out_dir, f"{base_name}_astrometry_solution.json")

            if os.path.exists(json_filename):
                with open(json_filename, "r") as infile:
                    data = json.load(infile)
            else:
                data = {}

            data["sources"] = sources
            with open(json_filename, "w") as outfile:
                json.dump(data, outfile, indent=4)
            _report(f"Updated JSON with {len(sources)} sources")

            # Photometric calibration: ensure only one instance at a time
            gaia_column = gaia_filter_map.get(filter_name, "phot_g_mean_mag")
            with PHOTOMETRY_SEMAPHORE:
                _report(f"Running photometric calibration (Gaia column: {gaia_column})")
                perform_photometric_calibration(json_filename, gaia_filter_column=gaia_column, max_workers=max_workers)

        except Exception as e:
            _report(f"Error during source extraction or calibration: {e}", color="red")

    # -------------------------------------------------------------------
    # 6) Collect all Light frames into tasks
    # -------------------------------------------------------------------
    light_tasks = []
    for filter_name, files in sorted_categories["Light"].items():
        for f in files:
            light_path = os.path.join(light_folder, filter_name, f)
            light_tasks.append((light_path, filter_name))

    # -------------------------------------------------------------------
    # 7) Process Light frames in parallel
    # -------------------------------------------------------------------
    _report(f"Lights: 0/{light_count}", category="light", current=0, total=light_count, done=False)
    processed_lights = 0
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for (lp, fn) in light_tasks:
            futures.append(executor.submit(calibrate_and_annotate, lp, fn))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                _report(f"Error in processing a light frame: {exc}", color="red")
            processed_lights += 1
            _report(f"Lights: {processed_lights}/{light_count}",
                    category="light", current=processed_lights, total=light_count, done=False)

    _report(f"Lights: {light_count}/{light_count} (done)",
            category="light", current=light_count, total=light_count, done=True, color="green")
    _report("All processing complete!", color="green")


# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------
def create_master_bias(bias_folder, calibration_folder, progress_callback=None):
    bias_files = [f for f in os.listdir(bias_folder) if f.lower().endswith(('.fit', '.fits'))]
    if not bias_files:
        raise FileNotFoundError(f"No bias frames found in folder: {bias_folder}")

    bias_frames = []
    for i, bias_file in enumerate(bias_files, 1):
        bias_path = os.path.join(bias_folder, bias_file)
        with fits.open(bias_path) as hdul:
            bias_data = hdul[0].data.astype(np.float32)
            bias_frames.append(bias_data)
        if progress_callback:
            progress_callback(i)

    master_bias = calculate_median_frame(bias_frames)
    master_bias_path = os.path.join(calibration_folder, 'master_bias.fit')
    fits.writeto(master_bias_path, master_bias, overwrite=True)


def create_master_dark(dark_folder, calibration_folder, master_bias=None, progress_callback=None):
    dark_files = [f for f in os.listdir(dark_folder) if f.lower().endswith(('.fit', '.fits'))]
    if not dark_files:
        raise FileNotFoundError(f"No dark frames found in folder: {dark_folder}")

    dark_frames = []
    for i, dark_file in enumerate(dark_files, 1):
        dark_path = os.path.join(dark_folder, dark_file)
        with fits.open(dark_path) as hdul:
            dark_data = hdul[0].data.astype(np.float32)
            if master_bias is not None:
                dark_data -= master_bias
                dark_data = np.clip(dark_data, 0, None)
            dark_frames.append(dark_data)
        if progress_callback:
            progress_callback(i)

    master_dark = calculate_median_frame(dark_frames)
    master_dark = np.clip(master_dark, 0, None)
    master_dark_path = os.path.join(calibration_folder, 'master_dark.fit')
    fits.writeto(master_dark_path, master_dark, overwrite=True)


def create_master_flats_for_filters(flat_folder, calibration_folder, master_bias=None, progress_callback=None):
    filters = [f for f in os.listdir(flat_folder) if os.path.isdir(os.path.join(flat_folder, f))]
    current_count = 0
    for filter_name in filters:
        filter_path = os.path.join(flat_folder, filter_name)
        flat_files = [f for f in os.listdir(filter_path) if f.lower().endswith(('.fit', '.fits'))]
        if not flat_files:
            raise FileNotFoundError(f"No flat frames found in {filter_path}")

        flat_frames = []
        for flat_file in flat_files:
            flat_path = os.path.join(filter_path, flat_file)
            with fits.open(flat_path) as hdul:
                flat_data = hdul[0].data.astype(np.float32)
                if master_bias is not None:
                    flat_data -= master_bias
                normalized_flat = normalize_frame(flat_data)
                flat_frames.append(normalized_flat)

            current_count += 1
            if progress_callback:
                progress_callback(current_count)

        master_flat = calculate_median_frame(flat_frames)
        master_flat_path = os.path.join(calibration_folder, f"master_flat_{filter_name}.fit")
        fits.writeto(master_flat_path, master_flat, overwrite=True)


def load_master_frame(folder, master_filename):
    path = os.path.join(folder, f"{master_filename}.fit")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{master_filename}.fit file not found in {folder}.")
    with fits.open(path) as hdul:
        data = hdul[0].data
    return data


def calculate_median_frame(frames):
    stacked = np.stack(frames, axis=0)
    return np.median(stacked, axis=0)


def normalize_frame(frame):
    mean_val = np.mean(frame)
    return frame / (mean_val if mean_val != 0 else 1)
