# photometric_calibration.py
import json
import random
import time
import numpy as np

from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
import concurrent.futures

# -------------------------------------------
#  Configurable parameters
# -------------------------------------------
SEARCH_RADIUS = 10 * u.arcsec     # Radius for Gaia cone search
DEFAULT_MAX_WORKERS   = 10        # Fallback number of threads if not provided
MAX_TOTAL_MATCHES = 25            # Limit how many sources to use for calibration
SIGMA_THRESHOLD   = 2.0           # Sigma threshold for iterative clipping
MAX_ITER          = 3             # Max iterations for sigma clipping


# ----------------------------------------------------------------------
#  MAIN ENTRY: perform_photometric_calibration
# ----------------------------------------------------------------------
def perform_photometric_calibration(json_path, gaia_filter_column="phot_g_mean_mag", max_workers=None):
    """
    Queries Gaia for up to MAX_TOTAL_MATCHES sources from the given JSON,
    does a linear fit (y = m*x + b) with sigma clipping, and stores the
    resulting slope & intercept. Also applies the calibration to all
    sources in the JSON, adding 'calibrated_mag' to each.

    Parameters
    ----------
    json_path : str
        Path to the JSON file that contains "sources" with at least
        'ra', 'dec', 'daofind_mag'.
    gaia_filter_column : str
        Which Gaia column to use, e.g. 'phot_bp_mean_mag', 'phot_rp_mean_mag',
        or 'phot_g_mean_mag'.
    max_workers : int, optional
        Number of worker threads for Gaia queries. If None, defaults to DEFAULT_MAX_WORKERS.

    Returns
    -------
    (slope, intercept) or None if calibration failed or no matches found.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Calibration] Could not load JSON {json_path}: {e}")
        return None

    sources_list = data.get("sources", [])
    if not sources_list:
        print(f"[Calibration] No sources found in JSON. Aborting.")
        return None

    # 1) Shuffle and pick up to MAX_TOTAL_MATCHES for the Gaia query
    random.shuffle(sources_list)
    subset = sources_list[:MAX_TOTAL_MATCHES]

    # 2) Run Gaia queries in parallel
    matches = run_gaia_queries(subset, gaia_filter_column=gaia_filter_column, max_workers=max_workers)

    if not matches:
        print("[Calibration] No valid Gaia matches to build calibration. Aborting.")
        return None

    # 3) Extract arrays for the linear fit
    x = np.array([m["daofind_mag"] for m in matches])
    y = np.array([m["gaia_mag"]     for m in matches])

    valid_idx = (~np.isnan(x)) & (~np.isnan(y))
    x = x[valid_idx]
    y = y[valid_idx]

    if len(x) < 2:
        print("[Calibration] Not enough points to do a linear fit. Aborting.")
        return None

    # 4) Do a linear fit with sigma clipping
    slope, intercept, mask = linear_fit_with_sigma_clipping(x, y, SIGMA_THRESHOLD, MAX_ITER)

    if slope is None or intercept is None:
        print("[Calibration] Failed to compute linear calibration.")
        return None

    print(f"[Calibration] Computed linear fit: y = {slope:.3f}*x + {intercept:.3f}")

    # 5) Store slope & intercept in the JSON
    data["photometric_calibration"] = {
        "gaia_filter_column": gaia_filter_column,
        "slope": slope,
        "intercept": intercept
    }

    # 6) Compute 'calibrated_mag' for ALL sources
    for s in sources_list:
        dao_mag = s.get("daofind_mag")
        if dao_mag is not None and not np.isnan(dao_mag):
            s["calibrated_mag"] = float(slope * dao_mag + intercept)
        else:
            s["calibrated_mag"] = None

    # 7) Write updated JSON to disk
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    return (slope, intercept)


# -------------------------------------------
#  Gaia Query
# -------------------------------------------
def run_gaia_queries(sources_list, gaia_filter_column, max_workers=None):
    """
    Queries Gaia for each source in `sources_list`, returning only those
    that yield a valid `gaia_filter_column` magnitude. We also keep the
    daofind_mag from the source for the final fit.

    Parameters
    ----------
    sources_list : list
        List of source dictionaries.
    gaia_filter_column : str
        Gaia column to query.
    max_workers : int, optional
        Number of worker threads to use.

    Returns
    -------
    List of dict with keys: {"daofind_mag", "gaia_mag"}.
    """
    matches = []
    start_time = time.time()
    completed_count = 0
    total_futures = len(sources_list)
    workers = max_workers if max_workers is not None else DEFAULT_MAX_WORKERS

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(query_single_source, src, gaia_filter_column): src
                   for src in sources_list}

        for future in concurrent.futures.as_completed(futures):
            completed_count += 1
            result = future.result()
            if result is not None:
                matches.append(result)

            if completed_count % 5 == 0 or completed_count == total_futures:
                elapsed = time.time() - start_time
                avg_time = elapsed / completed_count
                est_total = total_futures * avg_time
                remaining = est_total - elapsed
                print(f"[Gaia Query] Completed {completed_count}/{total_futures}. "
                      f"Est. time remaining: {remaining:.1f}s. Matches: {len(matches)}")
    return matches


def query_single_source(source, gaia_filter_column):
    """
    Tries to query Gaia for a single source. If successful, returns
    {"daofind_mag", "gaia_mag"}. If not, returns None.
    """
    try:
        ra = source.get("ra")
        dec = source.get("dec")
        dao_mag = source.get("daofind_mag")

        if ra is None or dec is None or dao_mag is None:
            return None

        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
        job = Gaia.cone_search_async(coordinate=coord, radius=SEARCH_RADIUS)
        results = job.get_results()

        if len(results) == 0 or (gaia_filter_column not in results.colnames):
            return None

        if 'dist' in results.colnames:
            results.sort('dist')
        elif 'angular_distance' in results.colnames:
            results.sort('angular_distance')

        best = results[0]
        val = best[gaia_filter_column]

        if (val is np.ma.masked) or np.isnan(val):
            return None

        return {
            "daofind_mag": float(dao_mag),
            "gaia_mag": float(val)
        }

    except Exception as e:
        return None


# -------------------------------------------
#  Linear Fit with Sigma Clipping
# -------------------------------------------
def linear_fit_with_sigma_clipping(x, y, sigma_threshold, max_iter=3):
    """
    Fits y = slope*x + intercept, using iterative sigma clipping.
    Returns (slope, intercept, mask) or (None, None, None) on failure.
    """
    if len(x) < 2:
        return (None, None, None)

    mask = np.ones_like(x, dtype=bool)
    slope, intercept = None, None

    for _ in range(max_iter):
        slope, intercept = np.polyfit(x[mask], y[mask], deg=1)
        y_fit = slope * x + intercept
        residuals = y - y_fit

        std_resid = np.std(residuals[mask])
        new_mask = mask & (np.abs(residuals) < sigma_threshold * std_resid)

        if np.array_equal(new_mask, mask):
            break

        mask = new_mask

        if sum(mask) < 2:
            return (None, None, None)

    return (slope, intercept, mask)
