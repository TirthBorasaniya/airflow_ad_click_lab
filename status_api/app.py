"""
Pipeline status service: reports the latest Airflow run and the current champion model.
Runs as its own container instead of an Airflow task that never finishes.
"""

import json
import logging
import os
import time
from pathlib import Path

import requests
from flask import Flask, jsonify, redirect, render_template

logger = logging.getLogger(__name__)

AIRFLOW_API_URL = os.getenv("AIRFLOW_API_URL", "http://airflow-apiserver:8080")
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME", "airflow")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "airflow")
TARGET_DAG_ID = os.getenv("TARGET_DAG_ID", "ad_click_pipeline")
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "/artifacts"))
REQUEST_TIMEOUT_SECONDS = 10
TOKEN_LIFETIME_SECONDS = 3600
TOKEN_REFRESH_MARGIN_SECONDS = 60
IN_PROGRESS_STATES = ("queued", "running")

app = Flask(__name__)
token_cache_dict = {"token": None, "expires_at": 0.0}


def get_auth_token():
    """Return a cached Airflow API token, requesting a new one when it is about to expire."""
    if token_cache_dict["token"] and \
            time.time() < token_cache_dict["expires_at"] - TOKEN_REFRESH_MARGIN_SECONDS:
        return token_cache_dict["token"]
    try:
        response = requests.post(
            f"{AIRFLOW_API_URL}/auth/token",
            json={"username": AIRFLOW_USERNAME, "password": AIRFLOW_PASSWORD},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as err:
        logger.error("token request failed: %s", err)
        return None
    if response.status_code not in (200, 201):
        logger.error("authentication failed: %s %s", response.status_code, response.text)
        return None
    token_cache_dict["token"] = response.json().get("access_token")
    token_cache_dict["expires_at"] = time.time() + TOKEN_LIFETIME_SECONDS
    return token_cache_dict["token"]


def read_json(path: Path):
    """Return parsed JSON from path, or None if the file is missing or invalid."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def get_latest_run():
    """
    Fetch the most recent run of the target DAG from the Airflow REST API.

    Returns
    -------
    run_dict : dict
        state, run_id, logical_date, start_date, end_date, and note.
    """
    run_dict = {"state": "unknown", "run_id": "N/A", "logical_date": "N/A",
                "start_date": "N/A", "end_date": "N/A", "note": ""}
    token = get_auth_token()
    if not token:
        run_dict["note"] = "could not authenticate with Airflow"
        return run_dict
    try:
        response = requests.get(
            f"{AIRFLOW_API_URL}/api/v2/dags/{TARGET_DAG_ID}/dagRuns",
            params={"order_by": "-run_after", "limit": 1},
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as err:
        run_dict["note"] = f"network error: {err}"
        return run_dict
    if response.status_code != 200:
        run_dict["note"] = f"Airflow API returned {response.status_code}"
        return run_dict

    run_list = response.json().get("dag_runs", [])
    if not run_list:
        run_dict["note"] = "no runs yet; trigger the DAG in Airflow"
        return run_dict
    run = run_list[0]
    run_dict.update({
        "state": run.get("state", "unknown"),
        "run_id": run.get("dag_run_id", "N/A"),
        "logical_date": run.get("logical_date") or run.get("run_after", "N/A"),
        "start_date": run.get("start_date") or "N/A",
        "end_date": run.get("end_date") or "N/A",
    })
    if run_dict["state"] in IN_PROGRESS_STATES:
        run_dict["note"] = "pipeline is still in progress"
    return run_dict


def build_status() -> dict:
    """Combine the latest run, run summary, champion record, and latest batch scoring."""
    return {
        "latest_run": get_latest_run(),
        "summary": read_json(ARTIFACT_DIR / "latest_summary.json"),
        "champion": read_json(ARTIFACT_DIR / "registry" / "champion.json"),
        "scoring": read_json(ARTIFACT_DIR / "latest_scoring.json"),
    }


@app.route("/")
def index():
    """Redirect to the success or failure page depending on the latest run."""
    o_success = get_latest_run()["state"] == "success"
    return redirect("/success" if o_success else "/failure")


@app.route("/success")
def success():
    """Render the success page with run and champion details."""
    return render_template("success.html", **build_status())


@app.route("/failure")
def failure():
    """Render the failure or in-progress page with the reason."""
    return render_template("failure.html", **build_status())


@app.route("/api/status")
def api_status():
    """Return the full status as JSON."""
    return jsonify(build_status())


@app.route("/health")
def health():
    """Liveness check for Docker."""
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5555")))
