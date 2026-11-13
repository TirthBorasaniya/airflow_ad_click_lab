"""
Ad click prediction pipeline: validate data, train candidate models in parallel, select the
best, and promote it only if it passes the quality gate.

Flow:
    load_and_validate -> check_data -> reject_data -> publish_summary
    check_data -> split_data -> train_candidate (mapped, one per candidate) -> select_best
    select_best -> quality_gate -> promote_model or keep_champion -> publish_summary

promote_model updates the champion model Asset, which triggers the ad_click_scoring DAG.
"""

import json
from datetime import timedelta

import pendulum
from airflow.sdk import Param, dag, task

from src import config, registry
from src.assets import CHAMPION_MODEL_ASSET
from src.data import load_dataset, validate_dataset
from src.notifications import (
    dag_failure_callback,
    dag_success_callback,
    send_notification,
    task_failure_callback,
)
from src.training import save_splits, score_on_test, select_best as pick_best, split_dataset
from src.training import train_candidate as fit_candidate

DECISION_FILE = "decision.json"
BAD_DATA_ROWS = 100
SUMMARY_FILE = "latest_summary.json"

# tasks returning a dict set multiple_outputs=False: with a dict annotation Airflow would push
# each key as its own XCom, and Airflow 3.0.2 rejects a None value pushed that way
DEFAULT_ARG_DICT = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(seconds=20),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=10),
    "on_failure_callback": task_failure_callback,
}


def write_decision(run_id: str, decision_dict: dict) -> None:
    """Store this run's outcome where publish_summary can read it."""
    (config.run_dir(run_id) / DECISION_FILE).write_text(json.dumps(decision_dict, indent=2))


@dag(
    dag_id="ad_click_pipeline",
    description="Validate, train candidates in parallel, and promote the best model behind a gate",
    schedule="@daily",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARG_DICT,
    params={
        "min_test_auc": Param(config.DEFAULT_MIN_TEST_AUC, type="number", minimum=0, maximum=1,
                              description="Floor every promoted model must reach"),
        "min_auc_gain": Param(config.DEFAULT_MIN_AUC_GAIN, type="number", minimum=0, maximum=1,
                              description="Required improvement over the current champion"),
        "simulate_bad_data": Param(False, type="boolean",
                                   description="Use only 100 rows to demonstrate data rejection"),
    },
    on_success_callback=dag_success_callback,
    on_failure_callback=dag_failure_callback,
    tags=["mlops", "lab2"],
    doc_md=__doc__,
)
def ad_click_pipeline():
    """Define the pipeline's tasks and their dependencies."""

    @task(multiple_outputs=False)
    def load_and_validate(params=None, run_id=None) -> dict:
        """Load the dataset, validate it, and save the validation report with the run."""
        df = load_dataset()
        if params and params.get("simulate_bad_data"):
            df = df.head(BAD_DATA_ROWS)
        report_dict = validate_dataset(df)
        (config.run_dir(run_id) / "validation.json").write_text(json.dumps(report_dict, indent=2))
        print(f"validation: {report_dict}")
        return report_dict

    @task.branch
    def check_data(report_dict: dict) -> str:
        """Continue to training only if the data passed validation."""
        return "split_data" if report_dict["valid"] else "reject_data"

    @task
    def reject_data(report_dict: dict, run_id=None) -> None:
        """Record and announce why this run's data was rejected."""
        reason = "; ".join(report_dict["error_list"])
        write_decision(run_id, {"decision": "data_rejected", "reason": reason})
        send_notification("Airflow pipeline: data rejected", f"run={run_id}, {reason}")

    @task
    def split_data(report_dict: dict, run_id=None) -> str:
        """Split the validated data and save the splits in this run's folder."""
        split_dict = split_dataset(load_dataset())
        return save_splits(split_dict, config.run_dir(run_id) / "splits.pkl")

    @task(multiple_outputs=False)
    def train_candidate(candidate: str, splits_path: str, run_id=None) -> dict:
        """Train one candidate; mapped so all candidates train in parallel."""
        result_dict = fit_candidate(candidate, splits_path, config.run_dir(run_id))
        print(f"{candidate}: validation {result_dict['validation']}")
        return result_dict

    @task(multiple_outputs=False)
    def select_best(result_list, splits_path: str, run_id=None) -> dict:
        """Pick the best candidate on validation AUC, then score it once on the test set."""
        result_list = list(result_list)
        best_dict = dict(pick_best(result_list))
        best_dict["test"] = score_on_test(best_dict["model_path"], splits_path)

        champion_dict = registry.load_champion(config.registry_dir())
        best_dict["champion_run_id"] = champion_dict["run_id"] if champion_dict else None
        # re-score the champion on this run's test set so both models face the same data
        best_dict["champion_test_auc"] = (
            score_on_test(champion_dict["model_path"], splits_path)["auc"]
            if champion_dict else None
        )
        leaderboard_list = sorted(result_list, key=lambda r: -r["validation"]["auc"])
        (config.run_dir(run_id) / "leaderboard.json").write_text(
            json.dumps(leaderboard_list, indent=2)
        )
        print(f"best: {best_dict['candidate']}, test {best_dict['test']}")
        return best_dict

    @task.branch
    def quality_gate(best_dict: dict, params=None, run_id=None) -> str:
        """Promote only if the candidate clears the floor and beats the champion."""
        o_promote, reason = registry.decide_promotion(
            best_dict["test"]["auc"], best_dict["champion_test_auc"],
            float(params["min_test_auc"]), float(params["min_auc_gain"]),
        )
        print(f"gate: {'promote' if o_promote else 'keep champion'} ({reason})")
        (config.run_dir(run_id) / "gate.json").write_text(
            json.dumps({"promote": o_promote, "reason": reason})
        )
        return "promote_model" if o_promote else "keep_champion"

    @task(outlets=[CHAMPION_MODEL_ASSET])
    def promote_model(best_dict: dict, run_id=None) -> None:
        """Register the candidate as the new champion, which schedules the scoring DAG."""
        gate_dict = json.loads((config.run_dir(run_id) / "gate.json").read_text())
        champion_dict = registry.promote(best_dict, config.registry_dir(), run_id)
        record_dict = {"decision": "promoted", "reason": gate_dict["reason"],
                       "candidate": best_dict["candidate"], "test_auc": best_dict["test"]["auc"],
                       "champion_test_auc": best_dict["champion_test_auc"], "run_id": run_id}
        registry.append_history(config.registry_dir(), record_dict)
        write_decision(run_id, record_dict)
        print(f"promoted {champion_dict['candidate']} from run {run_id}")

    @task
    def keep_champion(best_dict: dict, run_id=None) -> None:
        """Record that the current champion stays in place."""
        gate_dict = json.loads((config.run_dir(run_id) / "gate.json").read_text())
        record_dict = {"decision": "kept_champion", "reason": gate_dict["reason"],
                       "candidate": best_dict["candidate"], "test_auc": best_dict["test"]["auc"],
                       "champion_test_auc": best_dict["champion_test_auc"], "run_id": run_id}
        registry.append_history(config.registry_dir(), record_dict)
        write_decision(run_id, record_dict)

    @task(trigger_rule="none_failed_min_one_success", multiple_outputs=False)
    def publish_summary(run_id=None) -> dict:
        """Write the run outcome and current champion for the status service to read."""
        decision_dict = json.loads((config.run_dir(run_id) / DECISION_FILE).read_text())
        summary_dict = {
            "run_id": run_id,
            "decision": decision_dict,
            "champion": registry.load_champion(config.registry_dir()),
            "published_at": pendulum.now("UTC").isoformat(),
        }
        (config.ARTIFACT_DIR / SUMMARY_FILE).write_text(json.dumps(summary_dict, indent=2))
        return summary_dict

    report = load_and_validate()
    branch = check_data(report)
    rejected = reject_data(report)
    splits = split_data(report)
    branch >> [splits, rejected]

    results = train_candidate.partial(splits_path=splits).expand(candidate=config.CANDIDATE_NAMES)
    best = select_best(results, splits)
    gate = quality_gate(best)
    promoted = promote_model(best)
    kept = keep_champion(best)
    gate >> [promoted, kept]

    [promoted, kept, rejected] >> publish_summary()


ad_click_pipeline()
