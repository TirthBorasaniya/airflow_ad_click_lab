"""
DAG integrity tests: the DAG imports cleanly and has the expected structure.
Skipped automatically when Airflow is not installed.
"""

from pathlib import Path

import pytest

pytest.importorskip("airflow")

from airflow.models import DagBag  # noqa: E402

DAG_FOLDER = Path(__file__).resolve().parent.parent / "dags"
DAG_ID = "ad_click_pipeline"
SCORING_DAG_ID = "ad_click_scoring"
EXPECTED_TASK_SET = {
    "load_and_validate", "check_data", "reject_data", "split_data", "train_candidate",
    "select_best", "quality_gate", "promote_model", "keep_champion", "publish_summary",
}


@pytest.fixture(scope="module")
def dag_bag():
    return DagBag(dag_folder=str(DAG_FOLDER), include_examples=False)


def test_no_import_errors(dag_bag):
    assert dag_bag.import_errors == {}


def test_dag_structure(dag_bag):
    dag = dag_bag.get_dag(DAG_ID)
    assert dag is not None
    assert set(dag.task_ids) == EXPECTED_TASK_SET
    assert dag.max_active_runs == 1
    assert not dag.catchup


def test_branches_point_at_real_tasks(dag_bag):
    dag = dag_bag.get_dag(DAG_ID)
    assert dag.get_task("check_data").downstream_task_ids == {"split_data", "reject_data"}
    assert dag.get_task("quality_gate").downstream_task_ids == {"promote_model", "keep_champion"}


def test_default_args_set_retries_and_timeouts(dag_bag):
    task = dag_bag.get_dag(DAG_ID).get_task("split_data")
    assert task.retries == 2
    assert task.retry_exponential_backoff
    assert task.execution_timeout is not None


def test_summary_runs_after_any_outcome(dag_bag):
    task = dag_bag.get_dag(DAG_ID).get_task("publish_summary")
    assert task.upstream_task_ids == {"promote_model", "keep_champion", "reject_data"}
    assert task.trigger_rule == "none_failed_min_one_success"


def test_params_have_defaults(dag_bag):
    param_dict = dag_bag.get_dag(DAG_ID).params
    assert param_dict["min_test_auc"] == 0.9
    assert param_dict["min_auc_gain"] == 0.002
    assert param_dict["simulate_bad_data"] is False


def test_promotion_updates_the_champion_asset(dag_bag):
    outlet_list = dag_bag.get_dag(DAG_ID).get_task("promote_model").outlets
    assert [outlet.name for outlet in outlet_list] == ["ad_click_champion_model"]
    assert not dag_bag.get_dag(DAG_ID).get_task("keep_champion").outlets


def test_scoring_dag_is_scheduled_on_the_asset(dag_bag):
    dag = dag_bag.get_dag(SCORING_DAG_ID)
    assert dag is not None
    assert set(dag.task_ids) == {"validate_batch", "score_batch", "publish_scoring_summary"}
    assert type(dag.timetable).__name__ == "AssetTriggeredTimetable"


def test_dict_tasks_push_a_single_xcom(dag_bag):
    dict_task_dict = {
        DAG_ID: ["load_and_validate", "train_candidate", "select_best", "publish_summary"],
        SCORING_DAG_ID: ["validate_batch", "score_batch"],
    }
    for dag_id, task_id_list in dict_task_dict.items():
        dag = dag_bag.get_dag(dag_id)
        for task_id in task_id_list:
            assert dag.get_task(task_id).multiple_outputs is False, (dag_id, task_id)
