"""
Configuration for the ad click pipeline: paths, columns, validation thresholds, and candidates.
"""

import os
import re
from pathlib import Path

ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "/opt/airflow/artifacts"))
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "advertising.csv"
BATCH_PATH = Path(__file__).resolve().parent.parent / "data" / "new_visitors.csv"
ID_COLUMN = "visitor_id"

TARGET_COLUMN = "Clicked on Ad"
CONTINUOUS_COLUMNS = ["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage"]
BINARY_COLUMNS = ["Male"]
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + BINARY_COLUMNS

# validation thresholds
MIN_ROWS = 500
MIN_CLASS_FRACTION = 0.2
VALUE_RANGE_DICT = {
    "Daily Time Spent on Site": (0, 24 * 60),
    "Age": (0, 120),
    "Area Income": (0, None),
    "Daily Internet Usage": (0, 24 * 60),
}

# splitting and training
RANDOM_STATE = 42
TEST_FRACTION = 0.2
VALIDATION_FRACTION = 0.2
CANDIDATE_NAMES = ["logistic_regression", "random_forest", "gradient_boosting"]

# quality gate defaults; overridable per run through DAG params
DEFAULT_MIN_TEST_AUC = 0.9
DEFAULT_MIN_AUC_GAIN = 0.002

# probability above which a visitor is predicted to click
DECISION_THRESHOLD = 0.5


def safe_name(run_id: str) -> str:
    """Turn an Airflow run ID into a string that is safe as a folder name."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)


def run_dir(run_id: str, artifact_dir: Path = None) -> Path:
    """Return, creating it if needed, the artifact folder for one DAG run."""
    path = (artifact_dir or ARTIFACT_DIR) / "runs" / safe_name(run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def registry_dir(artifact_dir: Path = None) -> Path:
    """Return, creating it if needed, the model registry folder."""
    path = (artifact_dir or ARTIFACT_DIR) / "registry"
    path.mkdir(parents=True, exist_ok=True)
    return path
