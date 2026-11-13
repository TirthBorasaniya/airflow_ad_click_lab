"""
Airflow Assets shared between DAGs: the training DAG updates them, the scoring DAG waits on them.
"""

from airflow.sdk import Asset

# updated only when a new model is promoted; schedules the scoring DAG
CHAMPION_MODEL_ASSET = Asset("ad_click_champion_model")
