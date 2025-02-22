import argparse
import json
import random
import time
import numpy as np
import matplotlib.pyplot as plt

from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
import concurrent.futures

# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------
json_path = '/Users/orlando/Documents/GitHub/PHYS450-Orlando/demo small/Output 1/calibrated/B/calibrated_Light_M31_180.0s_Bin1_B_20231015-004727_0003_astrometry_solution.json'

search_radius = 10 * u.arcsec    
max_workers = 10                 
max_total_matches = 200         

PASTEL_COLORS = [
    "#FFB3BA",  #  pink
    "#BFFCC6",  #  green
    "#FFDFBA",  #  orange
    "#B4AEE8",  #  purple
]

POINT_COLOR = "#006766"

SIGMA_THRESHOLD = 2 


# -----------------------------------------------------------------------------
# GAIA QUERY FUNCTIONS
# -----------------------------------------------------------------------------
def query_source(source):
    try:
        ra = source.get("ra")
        dec = source.get("dec")
        daofind_mag = source.get("daofind_mag")

        # If RA/Dec missing, skip
        if ra is None or dec is None:
            return None

        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
        job = Gaia.cone_search_async(coordinate=coord, radius=search_radius)
        results = job.get_results()

        if len(results) == 0 or 'phot_bp_mean_mag' not in results.colnames:
            return None

        if 'dist' in results.colnames:
            results.sort('dist')
        elif 'angular_distance' in results.colnames:
            results.sort('angular_distance')

        best = results[0]
        val = best['phot_bp_mean_mag']

        # If masked or NaN, skip
        if (val is np.ma.masked) or np.isnan(val):
            return None

        # If valid, store in source
        gaia_bp_mag = float(val)
        source['gaia_bp_mag'] = gaia_bp_mag

        return {"daofind_mag": daofind_mag, "gaia_bp_mag": gaia_bp_mag}

    except Exception as e:
        print(f"Exception in query_source (RA={source.get('ra')}, DEC={source.get('dec')}): {e}")
        return None

def run_gaia_queries(sources_list):

    matches = []
    start_time = time.time()
    completed_count = 0
    total_futures = len(sources_list)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(query_source, src): src for src in sources_list}
        for future in concurrent.futures.as_completed(futures):
            completed_count += 1
            result = future.result()

            if result is not None:
                matches.append(result)

            if completed_count % 100 == 0 or completed_count == total_futures:
                elapsed = time.time() - start_time
                avg_time = elapsed / completed_count
                estimated_total = total_futures * avg_time
                remaining = estimated_total - elapsed
                print(f"Completed {completed_count}/{total_futures} queries. "
                      f"Est. time remaining: {remaining:.2f} s. "
                      f"Sources matched: {len(matches)}")

            if len(matches) >= max_total_matches:
                print(f"Reached {max_total_matches} matches. Stopping queries.")
                break

        # Cancel any unfinished tasks
        for fut in futures:
            if not fut.done():
                fut.cancel()

    return matches


# -----------------------------------------------------------------------------
# LOAD MATCHES FROM JSON
# -----------------------------------------------------------------------------
def get_matches_from_json(astrometry_data):

    matches = []
    for source in astrometry_data.get("sources", []):
        if "gaia_bp_mag" in source and "daofind_mag" in source:
            gaia_val = source["gaia_bp_mag"]
            dao_val = source["daofind_mag"]
            if gaia_val is not None and not np.isnan(gaia_val):
                matches.append({"daofind_mag": dao_val, "gaia_bp_mag": gaia_val})
    return matches


# -----------------------------------------------------------------------------
# OUTLIER REMOVAL (Iterative Sigma Clipping for Polynomial Fits)
# -----------------------------------------------------------------------------
def sigma_clip_polyfit(x, y, degree=1, sigma_threshold=3.0, max_iter=3, verbose=False):

    mask = np.ones_like(x, dtype=bool)
    for iteration in range(max_iter):
        coefs = np.polyfit(x[mask], y[mask], degree)

        y_fit = np.polyval(coefs, x)
        residuals = y - y_fit

        std_resid = np.std(residuals[mask])

        new_mask = mask & (np.abs(residuals) < sigma_threshold * std_resid)

        if np.array_equal(new_mask, mask):
            if verbose:
                print(f"Sigma-clip converged at iteration {iteration}")
            break

        mask = new_mask

        if verbose:
            print(f"Iteration {iteration+1}: kept {mask.sum()}/{len(x)} points.")

    return coefs, mask


# -----------------------------------------------------------------------------
# MAIN SCRIPT
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Calibration fits + outlier clipping, with sigma lines shown.")
    parser.add_argument('-g', '--gaia', action='store_true',
                        help="Query Gaia for matches; update JSON with Gaia match values.")
    args = parser.parse_args()

    with open(json_path, 'r') as f:
        astrometry_data = json.load(f)

    if args.gaia:
        sources_list = astrometry_data.get("sources", [])
        random.shuffle(sources_list)
        new_matches = run_gaia_queries(sources_list)

        with open(json_path, 'w') as f:
            json.dump(astrometry_data, f, indent=4)
        print(f"Updated JSON saved to: {json_path}")

    matches = get_matches_from_json(astrometry_data)
    if len(matches) == 0:
        print("No valid Gaia matches found in JSON. (Or all are NaN.) Aborting.")
        return

    if len(matches) > max_total_matches:
        matches = matches[:max_total_matches]

    x = np.array([m["daofind_mag"] for m in matches])
    y = np.array([m["gaia_bp_mag"] for m in matches])

    valid_idx = (~np.isnan(x)) & (~np.isnan(y))
    x = x[valid_idx]
    y = y[valid_idx]

    if len(x) < 2:
        print("Not enough points to fit after removing NaNs. Exiting.")
        return

    print(f"Using {len(x)} matched points for fitting.")
    print(f"x range: {x.min()} to {x.max()}")
    print(f"y range: {y.min()} to {y.max()}")


    fit_degrees = {
        'Linear': 1,
        'Quadratic': 2,
        'Cubic': 3,
        'Quartic': 4,
    }
    fit_results = {}

    for model_name, deg in fit_degrees.items():
        coefs, mask = sigma_clip_polyfit(
            x, y,
            degree=deg,
            sigma_threshold=SIGMA_THRESHOLD, 
            max_iter=3,
            verbose=False
        )
        y_fit_masked = np.polyval(coefs, x[mask])
        chi2 = np.sum((y[mask] - y_fit_masked)**2)

        fit_results[model_name] = {
            "coefs": coefs, 
            "mask": mask,       
            "chi2": chi2
        }


    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plt.subplots_adjust(hspace=1)

    axes = axes.flatten()

    x_plot = np.linspace(np.min(x), np.max(x), 200)

    for ax, (model_name, deg), color in zip(axes, fit_degrees.items(), PASTEL_COLORS):
        result = fit_results[model_name]
        coefs = result["coefs"]
        mask = result["mask"]
        chi2 = result["chi2"]

        y_plot = np.polyval(coefs, x_plot)

        ax.scatter(x[mask], y[mask], c=POINT_COLOR, s=12, alpha=0.9, label='Used Points')

        outliers = ~mask
        if np.any(outliers):
            ax.scatter(x[outliers], y[outliers], c='gray', s=12, alpha=0.5, label='Outliers')

        ax.plot(x_plot, y_plot, color=color, lw=2.5, label=f"{model_name} Fit")

        y_fit_kept = np.polyval(coefs, x[mask])
        residuals_kept = y[mask] - y_fit_kept
        std_resid = np.std(residuals_kept)

        # Plot 1σ, 2σ, 3σ lines in dotted style around best-fit
        for n_sigma in [1, 2, 3]:
            offset = n_sigma * std_resid
            upper = y_plot + offset
            lower = y_plot - offset
            ax.plot(x_plot, upper, ls='--', color=color, alpha=0.8, lw=1,
                    label=f"{n_sigma}σ" if n_sigma == 1 else None)
            ax.plot(x_plot, lower, ls='--', color=color, alpha=0.8, lw=1)

        coeff_str = np.round(coefs, 3).tolist()

        ax.set_title(f"{model_name}"
                     f"Chi²={chi2:.2f}, Coeffs={coeff_str}")
        ax.set_xlabel("Camera Magnitude")
        ax.set_ylabel("Gaia Magnitude")
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
