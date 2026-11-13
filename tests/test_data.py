"""
Tests for dataset loading and validation.
"""

import pandas as pd
import pytest

from src import config
from src.data import load_dataset, validate_dataset


@pytest.fixture
def real_df():
    """The dataset shipped with the lab."""
    return load_dataset()


def test_real_dataset_is_valid(real_df):
    report_dict = validate_dataset(real_df)
    assert report_dict["valid"], report_dict["error_list"]
    assert report_dict["row_count"] == 1000
    assert report_dict["positive_rate"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "mutate, expected_fragment",
    [
        (lambda df: df.drop(columns=["Age"]), "missing columns"),
        (lambda df: df.head(100), "only 100 rows"),
        (lambda df: df.assign(Age=df["Age"].where(df.index > 5)), "missing values"),
        (lambda df: df.assign(**{config.TARGET_COLUMN: df[config.TARGET_COLUMN] * 2}),
         "target must be 0 or 1"),
        (lambda df: df.assign(**{config.TARGET_COLUMN: (df.index < 50).astype(int)}),
         "minority class"),
        (lambda df: df.assign(Male=df["Male"] + 1), "Male must be 0 or 1"),
        (lambda df: df.assign(Age=df["Age"].where(df.index > 0, -5)), "out of range"),
    ],
)
def test_invalid_datasets_are_rejected(real_df, mutate, expected_fragment):
    report_dict = validate_dataset(mutate(real_df.copy()))
    assert not report_dict["valid"]
    assert any(expected_fragment in error for error in report_dict["error_list"]), \
        report_dict["error_list"]


def test_empty_dataframe_reports_errors_without_crashing():
    report_dict = validate_dataset(pd.DataFrame())
    assert not report_dict["valid"]
    assert report_dict["row_count"] == 0
