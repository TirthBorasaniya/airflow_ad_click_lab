"""
Batch scoring DAG: scores new visitors with the champion model.

Scheduled on the champion model Asset rather than a clock, so it runs automatically each time the
training DAG promotes a new model. It can also be triggered manually.

Flow:
    validate_batch -> score_batch -> publish_scoring_summary
"""

import json
from datetime import timedelta

import pendulum
from airflow.sdk import dag, task

from src import config, registry
from src.assets import CHAMPION_MODEL_ASSET
from src.data import load_dataset, validate_batch as check_batch
from src.notifications import dag_failure_callback, task_failure_callback
from src.scoring import load_model, score_batch as score_rows, summarize_scores

SCORING_SUMMARY_FILE = "latest_scoring.json"

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


@dag(
    dag_id="ad_click_scoring",
    description="Score new visitors whenever a new champion model is promoted",
    schedule=[CHAMPION_MODEL_ASSET],
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARG_DICT,
    on_failure_callback=dag_failure_callback,
    tags=["mlops", "lab2"],
    doc_md=__doc__,
)
def ad_click_scoring():
    """Define the scoring tasks and their dependencies."""

    @task(multiple_outputs=False)
    def validate_batch() -> dict:
        """Check the incoming batch; fail the run with the reasons if it cannot be scored."""
        report_dict = check_batch(load_dataset(config.BATCH_PATH))
        print(f"batch validation: {report_dict}")
        if not report_dict["valid"]:
            raise ValueError(f"batch rejected: {'; '.join(report_dict['error_list'])}")
        return report_dict

    @task(multiple_outputs=False)
    def score_batch(report_dict: dict, run_id=None) -> dict:
        """Score the batch with the current champion and save the predictions."""
        champion_dict = registry.load_champion(config.registry_dir())
        if champion_dict is None:
            raise ValueError("no champion model; run ad_click_pipeline first")

        scored_df = score_rows(load_model(champion_dict["model_path"]),
                               load_dataset(config.BATCH_PATH))
        output_dir = config.ARTIFACT_DIR / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{config.safe_name(run_id)}.csv"
        scored_df.to_csv(output_path, index=False)

        summary_dict = {
            **summarize_scores(scored_df),
            "run_id": run_id,
            "champion_run_id": champion_dict["run_id"],
            "champion_candidate": champion_dict["candidate"],
            "predictions_path": str(output_path),
        }
        print(f"scoring: {summary_dict}")
        return summary_dict

    @task
    def publish_scoring_summary(summary_dict: dict) -> None:
        """Write the latest scoring summary for the status service to read."""
        summary_dict = {**summary_dict, "published_at": pendulum.now("UTC").isoformat()}
        (config.ARTIFACT_DIR / SCORING_SUMMARY_FILE).write_text(json.dumps(summary_dict, indent=2))

    publish_scoring_summary(score_batch(validate_batch()))


ad_click_scoring()
