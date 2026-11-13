"""
File-based model registry: the current champion, the promotion rule, and a decision history.
"""

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src import config

CHAMPION_FILE = "champion.json"
HISTORY_FILE = "history.jsonl"


def load_champion(registry_path: Path):
    """Return the champion record, or None if no model has been promoted yet."""
    champion_path = Path(registry_path) / CHAMPION_FILE
    if not champion_path.exists():
        return None
    return json.loads(champion_path.read_text())


def decide_promotion(candidate_auc: float, champion_auc, min_test_auc: float,
                     min_gain: float):
    """
    Decide whether the candidate replaces the champion.

    Both AUCs must come from the same test set.

    Parameters
    ----------
    candidate_auc : float
        Test AUC of the best candidate from this run.
    champion_auc : float or None
        Test AUC of the current champion on the same test set, or None if there is none.
    min_test_auc : float
        Absolute floor every promoted model must reach.
    min_gain : float
        Required improvement over the champion.

    Returns
    -------
    o_promote : bool
        True if the candidate should be promoted.
    reason : str
        Short explanation recorded with the decision.
    """
    if not math.isfinite(candidate_auc):
        return False, "candidate AUC is not a finite number"
    if candidate_auc < min_test_auc:
        return False, f"test AUC {candidate_auc:.4f} is below the floor of {min_test_auc}"
    if champion_auc is None:
        return True, "no current champion"
    gain = candidate_auc - champion_auc
    if gain >= min_gain:
        return True, f"test AUC improved by {gain:.4f} (required {min_gain})"
    return False, f"test AUC changed by {gain:+.4f}, below the required {min_gain}"


def append_history(registry_path: Path, record_dict: dict) -> None:
    """Append one decision record to the history file."""
    with open(Path(registry_path) / HISTORY_FILE, "a") as f:
        f.write(json.dumps(record_dict) + "\n")


def promote(best_dict: dict, registry_path: Path, run_id: str, now=None) -> dict:
    """
    Copy the best model into the registry and make it the champion.

    Parameters
    ----------
    best_dict : dict
        Output of the select_best task: candidate, model_path, validation, test.
    registry_path : pathlib.Path
        Registry folder.
    run_id : str
        Airflow run that produced the model.
    now : datetime.datetime or None
        Promotion time; defaults to the current UTC time.

    Returns
    -------
    champion_dict : dict
        The new champion record.
    """
    now = now or datetime.now(timezone.utc)
    model_dir = Path(registry_path) / "models" / config.safe_name(run_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    registered_path = model_dir / "model.pkl"
    shutil.copy2(best_dict["model_path"], registered_path)

    champion_dict = {
        "run_id": run_id,
        "candidate": best_dict["candidate"],
        "model_path": str(registered_path),
        "validation": best_dict["validation"],
        "test": best_dict["test"],
        "promoted_at": now.isoformat(),
    }
    (Path(registry_path) / CHAMPION_FILE).write_text(json.dumps(champion_dict, indent=2))
    return champion_dict
