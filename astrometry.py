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

# Polling configuration
MAX_POLL_ATTEMPTS = 60      # Increased maximum number of polling attempts
POLL_INTERVAL = 10          # Seconds between polling attempts

def setup_astrometry():
    """
    Logs into astrometry.net, checks available memory, and returns the session token.
    Raises an Exception if login fails.
    """
    available_memory = psutil.virtual_memory().available
    print("Available memory (bytes):", available_memory)
    
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
    Uploads an image to astrometry.net and polls until the astrometric solution is available.
    Raises an exception if a job ID is not assigned within the maximum polling attempts.
    """
    with requests.Session() as s:
        upload_payload = {
            "session": session,
            "publicly_visible": "n",          
            "allow_commercial_usage": "n",    
            "allow_modifications": "n"         
        }
        
        print("Uploading image:", image_path)
        with open(image_path, "rb") as f:
            files = {"file": f}
            data = {"request-json": json.dumps(upload_payload)}
            upload_response = s.post(UPLOAD_URL, data=data, files=files)
        
        upload_result = upload_response.json()
        print("Upload response:", upload_result)
        
        subid = upload_result.get("subid")
        if not subid:
            raise Exception("Image upload failed or did not return a submission id.")
        
        # --- Step 2: Poll for a job ID ---
        submission_url = f"{SUBMISSIONS_URL}/{subid}"
        job_id = None
        attempt = 0
        print("Polling for job ID...")
        while job_id is None and attempt < MAX_POLL_ATTEMPTS:
            time.sleep(POLL_INTERVAL)
            attempt += 1
            submission_response = s.get(submission_url)
            submission_data = submission_response.json()
            print(f"Submission status (attempt {attempt}):", submission_data)
            # Check if jobs list is non-empty and contains a valid job ID
            if "jobs" in submission_data:
                jobs = submission_data["jobs"]
                if jobs and jobs[0] is not None:
                    job_id = jobs[0]  # Use the first job id
                    break
        
        if job_id is None:
            raise Exception("Timeout waiting for job ID after {} attempts.".format(MAX_POLL_ATTEMPTS))
        print("Job ID assigned:", job_id)
        
        # --- Step 3: Poll for the astrometric solution ---
        job_info_url = f"{JOBS_URL}/{job_id}/info"
        calibration = None
        attempt = 0
        print("Polling for job completion and astrometric calibration...")
        while calibration is None and attempt < MAX_POLL_ATTEMPTS:
            time.sleep(POLL_INTERVAL)
            attempt += 1
            job_info_response = s.get(job_info_url)
            job_info = job_info_response.json()
            print(f"Job info (attempt {attempt}):", job_info)
            if "calibration" in job_info and job_info["calibration"]:
                calibration = job_info["calibration"]
                break
        
        if calibration is None:
            raise Exception("Timeout waiting for astrometric solution after {} attempts.".format(MAX_POLL_ATTEMPTS))
        
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

