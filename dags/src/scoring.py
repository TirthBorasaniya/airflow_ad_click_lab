"""
Batch scoring of new visitors with the champion model.
"""

import pickle

import pandas as pd

from src import config


def load_model(model_path):
    """Load a pickled scikit-learn pipeline."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def score_batch(model, batch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a batch and return one row per visitor.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Fitted pipeline that accepts raw FEATURE_COLUMNS.
    batch_df : pandas.DataFrame
        Validated batch with ID_COLUMN and FEATURE_COLUMNS.

    Returns
    -------
    scored_df : pandas.DataFrame
        ID_COLUMN, click_probability, and predicted_click (0 or 1).
    """
    probability = model.predict_proba(batch_df[config.FEATURE_COLUMNS])[:, 1]
    return pd.DataFrame({
        config.ID_COLUMN: batch_df[config.ID_COLUMN].to_numpy(),
        "click_probability": probability.round(4),
        "predicted_click": (probability >= config.DECISION_THRESHOLD).astype(int),
    })


def summarize_scores(scored_df: pd.DataFrame) -> dict:
    """Return the row count, predicted click rate, and mean probability of a scored batch."""
    return {
        "rows_scored": len(scored_df),
        "predicted_click_rate": round(float(scored_df["predicted_click"].mean()), 4),
        "mean_click_probability": round(float(scored_df["click_probability"].mean()), 4),
    }
