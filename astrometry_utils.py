import requests
import time
import os
import json

ASTROMETRY_BASE_URL = "http://nova.astrometry.net/api/"
API_KEY = "pyplbxajvnsqyifn"

def solve_astrometry(image_path, output_json_path, api_key, attempts=30, sleep_time=10):
    """
    Solve astrometry for a given image using astrometry.net and save results to a JSON file.

    Parameters:
    -----------
    image_path : str
        The local path to the image (FITS, JPEG, etc.) to be uploaded for astrometry.
    output_json_path : str
        Where the JSON output with astrometry details will be saved.
    api_key : str
        The astrometry.net API key.
    attempts : int
        How many times to poll for a job result before giving up.
    sleep_time : int
        How many seconds to wait between status checks.
    """
    # Step 1: Login to get a session
    session_token = login(api_key)
    
    # Step 2: Upload the image
    sub_id = upload_image(session_token, image_path)

    # Step 3: Poll submission until a job is created
    job_id = None
    for i in range(attempts):
        submission_status = check_submission(sub_id)
        jobs = submission_status.get("jobs", [])
        if jobs:
            job_id = jobs[0]  # Take the first job ID
            break
        time.sleep(sleep_time)

    if job_id is None:
        raise RuntimeError(f"No job ID returned after {attempts} attempts.")

    # Step 4: Poll the job until it's complete
    job_status = None
    for i in range(attempts):
        job_status = check_job(job_id)
        if job_status.get("status") == "success":
            break
        if job_status.get("status") == "failure":
            raise RuntimeError(f"Astrometry job {job_id} failed.")
        time.sleep(sleep_time)

    # Step 5: Retrieve final astrometry data
    job_details = get_job_details(job_id)

    # Step 6: Save to JSON
    save_to_json(job_details, output_json_path)
    print(f"Astrometry details for {image_path} saved to {output_json_path}")


def login(api_key):
    """
    Login to astrometry.net to obtain a session token.
    """
    url = os.path.join(ASTROMETRY_BASE_URL, "login")
    data = {"apikey": api_key}
    response = requests.post(url, data=data)
    response.raise_for_status()
    json_resp = response.json()
    session_token = json_resp.get("session")
    if not session_token:
        raise RuntimeError(f"Login failed; response: {json_resp}")
    return session_token


def upload_image(session_token, image_path):
    """
    Upload an image for astrometric calibration.
    """
    url = os.path.join(ASTROMETRY_BASE_URL, "upload")
    files = {"file": open(image_path, "rb")}
    data = {"session": session_token}

    response = requests.post(url, data=data, files=files)
    response.raise_for_status()
    json_resp = response.json()
    sub_id = json_resp.get("subid")
    if sub_id is None:
        raise RuntimeError(f"Upload failed; response: {json_resp}")
    return sub_id


def check_submission(sub_id):
    """
    Check the status of a submission by submission ID.
    """
    url = os.path.join(ASTROMETRY_BASE_URL, "submissions", str(sub_id))
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def check_job(job_id):
    """
    Check job status (e.g., 'solving', 'success', or 'failure').
    """
    url = os.path.join(ASTROMETRY_BASE_URL, "jobs", str(job_id))
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def get_job_details(job_id):
    """
    Retrieve detailed astrometry results from a completed job.
    """
    url = os.path.join(ASTROMETRY_BASE_URL, "jobs", str(job_id))
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def save_to_json(data, output_path):
    """
    Save data (dictionary) to a JSON file.
    """
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
