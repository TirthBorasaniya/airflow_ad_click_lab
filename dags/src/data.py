"""
Loading and validation of the advertising dataset before any training happens.
"""

import pandas as pd

from src import config


def load_dataset(csv_path=config.DATA_PATH) -> pd.DataFrame:
    """Read the advertising CSV."""
    return pd.read_csv(csv_path)


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Check that a dataset is fit for training.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataset.

    Returns
    -------
    report_dict : dict
        valid (bool), row_count, error_list (list of str), and positive_rate (float or None).
    """
    error_list = []
    required_list = config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    missing_list = [c for c in required_list if c not in df.columns]
    if missing_list:
        error_list.append(f"missing columns: {missing_list}")

    if len(df) < config.MIN_ROWS:
        error_list.append(f"only {len(df)} rows, at least {config.MIN_ROWS} required")

    positive_rate = None
    present_list = [c for c in required_list if c in df.columns]
    null_count = int(df[present_list].isna().sum().sum())
    if null_count:
        error_list.append(f"{null_count} missing values in required columns")

    if config.TARGET_COLUMN in df.columns:
        label_set = set(df[config.TARGET_COLUMN].dropna().unique())
        if not label_set <= {0, 1}:
            error_list.append(f"target must be 0 or 1, found {sorted(label_set)}")
        elif len(df):
            positive_rate = float(df[config.TARGET_COLUMN].mean())
            minority = min(positive_rate, 1 - positive_rate)
            if minority < config.MIN_CLASS_FRACTION:
                error_list.append(
                    f"minority class is {minority:.1%}, at least "
                    f"{config.MIN_CLASS_FRACTION:.0%} required"
                )

    error_list.extend(feature_value_errors(df))

    return {
        "valid": not error_list,
        "row_count": len(df),
        "error_list": error_list,
        "positive_rate": positive_rate,
    }


def feature_value_errors(df: pd.DataFrame) -> list:
    """Return errors for binary columns that are not 0 or 1 and values outside allowed ranges."""
    error_list = []
    for column in config.BINARY_COLUMNS:
        if column in df.columns and not set(df[column].dropna().unique()) <= {0, 1}:
            error_list.append(f"{column} must be 0 or 1")

    for column, (low, high) in config.VALUE_RANGE_DICT.items():
        if column not in df.columns:
            continue
        values = df[column].dropna()
        out_of_range = (values < low) | (values > high if high is not None else False)
        if out_of_range.any():
            error_list.append(f"{int(out_of_range.sum())} values of {column} out of range")
    return error_list


def validate_batch(df: pd.DataFrame) -> dict:
    """
    Check that a batch of new visitors can be scored.

    Parameters
    ----------
    df : pandas.DataFrame
        Batch with ID_COLUMN and FEATURE_COLUMNS; no target column is needed.

    Returns
    -------
    report_dict : dict
        valid (bool), row_count, and error_list (list of str).
    """
    error_list = []
    required_list = [config.ID_COLUMN] + config.FEATURE_COLUMNS
    missing_list = [c for c in required_list if c not in df.columns]
    if missing_list:
        error_list.append(f"missing columns: {missing_list}")
    if df.empty:
        error_list.append("batch is empty")
    present_list = [c for c in required_list if c in df.columns]
    null_count = int(df[present_list].isna().sum().sum())
    if null_count:
        error_list.append(f"{null_count} missing values in required columns")
    if config.ID_COLUMN in df.columns and df[config.ID_COLUMN].duplicated().any():
        error_list.append(f"duplicate values in {config.ID_COLUMN}")
    error_list.extend(feature_value_errors(df))
    return {"valid": not error_list, "row_count": len(df), "error_list": error_list}
