"""
Tests for batch validation and batch scoring.
"""

import pytest

from src import config
from src.data import load_dataset, validate_batch
from src.scoring import load_model, score_batch, summarize_scores
from src.training import save_splits, split_dataset, train_candidate


@pytest.fixture(scope="module")
def batch_df():
    return load_dataset(config.BATCH_PATH)


def test_shipped_batch_is_valid(batch_df):
    report_dict = validate_batch(batch_df)
    assert report_dict["valid"], report_dict["error_list"]
    assert report_dict["row_count"] == 200
    assert config.TARGET_COLUMN not in batch_df.columns


@pytest.mark.parametrize(
    "mutate, expected_fragment",
    [
        (lambda df: df.drop(columns=["visitor_id"]), "missing columns"),
        (lambda df: df.head(0), "batch is empty"),
        (lambda df: df.assign(Age=df["Age"].where(df.index > 0)), "missing values"),
        (lambda df: df.assign(visitor_id="V0001"), "duplicate values"),
        (lambda df: df.assign(Male=2), "Male must be 0 or 1"),
    ],
)
def test_invalid_batches_are_rejected(batch_df, mutate, expected_fragment):
    report_dict = validate_batch(mutate(batch_df.copy()))
    assert not report_dict["valid"]
    assert any(expected_fragment in e for e in report_dict["error_list"]), report_dict


def test_score_batch_returns_one_row_per_visitor(batch_df, tmp_path):
    splits_path = save_splits(split_dataset(load_dataset()), tmp_path / "splits.pkl")
    result_dict = train_candidate("logistic_regression", splits_path, tmp_path)
    scored_df = score_batch(load_model(result_dict["model_path"]), batch_df)

    assert list(scored_df.columns) == [config.ID_COLUMN, "click_probability", "predicted_click"]
    assert scored_df[config.ID_COLUMN].tolist() == batch_df[config.ID_COLUMN].tolist()
    assert scored_df["click_probability"].between(0, 1).all()
    assert set(scored_df["predicted_click"]) <= {0, 1}
    assert ((scored_df["click_probability"] >= 0.5) == (scored_df["predicted_click"] == 1)).all()

    summary_dict = summarize_scores(scored_df)
    assert summary_dict["rows_scored"] == 200
    assert 0.3 < summary_dict["predicted_click_rate"] < 0.7
