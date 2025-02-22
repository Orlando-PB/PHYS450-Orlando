# photometric_calibration.py
import json
import random
import time
import numpy as np

from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
import concurrent.futures

#  Configurable parameters
CHUNK_SIZE          = 25         
GAIA_QUERY_TIMEOUT  = 10         
SEARCH_RADIUS       = 10 * u.arcsec     
DEFAULT_MAX_WORKERS = 4             
MAX_TOTAL_MATCHES   = 50        
SIGMA_THRESHOLD     = 2.0           
MAX_ITER            = 3            



def perform_photometric_calibration(json_path, gaia_filter_column="phot_g_mean_mag", max_workers=None):

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

    random.shuffle(sources_list)
    subset = sources_list[:MAX_TOTAL_MATCHES]

    matches = run_gaia_queries_in_chunks(subset,
                                         gaia_filter_column=gaia_filter_column,
                                         chunk_size=CHUNK_SIZE,
                                         gaia_timeout=GAIA_QUERY_TIMEOUT,
                                         max_workers=max_workers)

    if not matches:
        print("[Calibration] No valid Gaia matches to build calibration. Aborting.")
        return None

    x = np.array([m["daofind_mag"] for m in matches])
    y = np.array([m["gaia_mag"]     for m in matches])

    valid_idx = (~np.isnan(x)) & (~np.isnan(y))
    x = x[valid_idx]
    y = y[valid_idx]

    if len(x) < 2:
        print("[Calibration] Not enough points to do a linear fit. Aborting.")
        return None

    slope, intercept, mask = linear_fit_with_sigma_clipping(x, y, SIGMA_THRESHOLD, MAX_ITER)

    if slope is None or intercept is None:
        print("[Calibration] Failed to compute linear calibration.")
        return None

    print(f"[Calibration] Computed linear fit: y = {slope:.3f}*x + {intercept:.3f}")

    data["photometric_calibration"] = {
        "gaia_filter_column": gaia_filter_column,
        "slope": slope,
        "intercept": intercept
    }

    for s in sources_list:
        dao_mag = s.get("daofind_mag")
        if dao_mag is not None and not np.isnan(dao_mag):
            s["calibrated_mag"] = float(slope * dao_mag + intercept)
        else:
            s["calibrated_mag"] = None

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    return (slope, intercept)


def run_gaia_queries_in_chunks(sources_list, gaia_filter_column,
                               chunk_size=CHUNK_SIZE,
                               gaia_timeout=GAIA_QUERY_TIMEOUT,
                               max_workers=None):
   
    Gaia.TIMEOUT = gaia_timeout

    total_sources = len(sources_list)
    all_matches = []
    start_time = time.time()
    global_completed = 0 

    workers = max_workers if max_workers is not None else DEFAULT_MAX_WORKERS

    for i in range(0, total_sources, chunk_size):
        chunk = sources_list[i:i+chunk_size]

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_src = {
                executor.submit(query_single_source, src, gaia_filter_column): src
                for src in chunk
            }

            for future in concurrent.futures.as_completed(future_to_src):
                global_completed += 1
                src = future_to_src[future]
                try:
                    result = future.result(timeout=gaia_timeout)
                except concurrent.futures.TimeoutError:
                    result = None
                except Exception:
                    result = None

                if result is not None:
                    all_matches.append(result)

                if (global_completed % 5 == 0) or (global_completed == total_sources):
                    elapsed = time.time() - start_time
                    avg_time = elapsed / global_completed
                    est_total = total_sources * avg_time
                    remaining = est_total - elapsed
                    print(f"[Gaia Query] Completed {global_completed}/{total_sources}. "
                          f"Est. time remaining: {remaining:.1f}s. Matches: {len(all_matches)}")

    return all_matches



def query_single_source(source, gaia_filter_column):
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

        # Sort by distance if possible
        if 'dist' in results.colnames:
            results.sort('dist')
        elif 'angular_distance' in results.colnames:
            results.sort('angular_distance')

        best = results[0]
        val = best[gaia_filter_column]
        if (val is np.ma.masked) or np.isnan(val):
            return None

        return {"daofind_mag": float(dao_mag), "gaia_mag": float(val)}

    except Exception:
        return None



def linear_fit_with_sigma_clipping(x, y, sigma_threshold, max_iter=3):

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
