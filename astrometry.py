# astrometry_module.py

import requests
import json
import time
import psutil
import os

# --- Configuration ---
API_KEY = "pyplbxajvnsqyifn"  # Replace with your actual API key
BASE_LOGIN_URL = "http://nova.astrometry.net/api/login"
UPLOAD_URL = "http://nova.astrometry.net/api/upload"
SUBMISSIONS_URL = "http://nova.astrometry.net/api/submissions"
JOBS_URL = "http://nova.astrometry.net/api/jobs"


def setup_astrometry():
    """
    Logs into astrometry.net, checks available memory, and returns the session token.
    Raises an Exception if login fails.
    """
    # Check available memory (for informational purposes)
    available_memory = psutil.virtual_memory().available
    print("Available memory (bytes):", available_memory)
    
    # Log in to astrometry.net
    login_payload = {"apikey": API_KEY}
    response = requests.post(BASE_LOGIN_URL, data={"request-json": json.dumps(login_payload)})
    data = response.json()
    session = data.get("session")
    if not session:
        raise Exception("Login failed: {}".format(data))
    print("Logged in successfully. Session token:", session)
    return session


def process_image(image_path, session):
    """
    Processes a single image:
      - Uploads the image.
      - Polls for the job ID.
      - Polls until the astrometric solution is ready.
      - Saves the calibration data as a JSON file in the same directory as the image.
    
    Returns the astrometric solution data.
    
    Raises an Exception if any step fails.
    """
    # --- Step 1: Upload the image ---
    upload_payload = {
        "session": session,
        "publicly_visible": "y",          # Change to "n" if you do not wish the image to be public
        "allow_commercial_usage": "d",     # "y" for yes, "n" for no, "d" for default
        "allow_modifications": "d"         # same as above
    }
    
    print("Uploading image:", image_path)
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {"request-json": json.dumps(upload_payload)}
        upload_response = requests.post(UPLOAD_URL, data=data, files=files)
    
    upload_result = upload_response.json()
    print("Upload response:", upload_result)
    
    subid = upload_result.get("subid")
    if not subid:
        raise Exception("Image upload failed or did not return a submission id.")
    
    # --- Step 2: Poll for a job ID ---
    submission_url = f"{SUBMISSIONS_URL}/{subid}"
    job_id = None
    print("Polling for job ID...")
    while job_id is None:
        time.sleep(10)
        submission_response = requests.get(submission_url)
        submission_data = submission_response.json()
        print("Submission status:", submission_data)
        if "jobs" in submission_data and submission_data["jobs"]:
            job_id = submission_data["jobs"][0]  # Use the first job id
            break
    if not job_id:
        raise Exception("No job ID was returned.")
    print("Job ID assigned:", job_id)
    
    # --- Step 3: Poll for the astrometric solution ---
    job_info_url = f"{JOBS_URL}/{job_id}/info"
    calibration = None
    print("Polling for job completion and astrometric calibration...")
    while calibration is None:
        time.sleep(10)
        job_info_response = requests.get(job_info_url)
        job_info = job_info_response.json()
        print("Job info:", job_info)
        if "calibration" in job_info and job_info["calibration"]:
            calibration = job_info["calibration"]
            break
    if calibration is None:
        raise Exception("The job did not produce an astrometric solution.")
    
    # --- Step 4: Extract and display the calibration ---
    ra = calibration.get("ra")
    dec = calibration.get("dec")
    pixscale = calibration.get("pixscale")
    orientation = calibration.get("orientation")
    
    print("\nAstrometric solution:")
    print("Right Ascension (RA):", ra)
    print("Declination (DEC):", dec)
    print("Pixel Scale (arcsec/pixel):", pixscale)
    print("Orientation (degrees):", orientation)
    
    solution_data = {
        "Right Ascension": ra,
        "Declination": dec,
        "Pixel Scale (arcsec/pixel)": pixscale,
        "Orientation (degrees)": orientation,
        "Field Radius": calibration.get("radius"),
        "Parity": calibration.get("parity")
    }
    
    # --- Step 5: Save the solution as a JSON file next to the image ---
    image_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    json_filename = os.path.join(image_dir, f"{base_name}_astrometry_solution.json")
    
    with open(json_filename, "w") as outfile:
        json.dump(solution_data, outfile, indent=4)
    
    print("Astrometric solution saved to", json_filename)
    return solution_data


if __name__ == '__main__':
    # Example usage when running this module directly.
    # Replace the image path below with your own test image.
    image_path = "/Users/orlando/Documents/GitHub/PHYS450-Orlando/demo/Output 1/calibrated/B/calibrated_Light_M31_180.0s_Bin1_B_20231015-004727_0003.fit"
    session_token = setup_astrometry()
    process_image(image_path, session_token)
