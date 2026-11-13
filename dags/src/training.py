"""
Data splitting, candidate model training, and evaluation for the ad click pipeline.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config

CANDIDATE_FACTORY_DICT = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, random_state=config.RANDOM_STATE
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=config.RANDOM_STATE),
}


def split_dataset(df: pd.DataFrame) -> dict:
    """
    Split into stratified train, validation, and test sets.

    Returns
    -------
    split_dict : dict of str to tuple
        Maps "train", "validation", and "test" to (X, y) with FEATURE_COLUMNS.
    """
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=config.TEST_FRACTION, stratify=y, random_state=config.RANDOM_STATE
    )
    val_share = config.VALIDATION_FRACTION / (1 - config.TEST_FRACTION)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_share, stratify=y_rest, random_state=config.RANDOM_STATE
    )
    return {"train": (X_train, y_train), "validation": (X_val, y_val), "test": (X_test, y_test)}


def save_splits(split_dict: dict, output_path: Path) -> str:
    """Pickle the splits and return the file path as a string."""
    with open(output_path, "wb") as f:
        pickle.dump(split_dict, f)
    return str(output_path)


def load_splits(splits_path) -> dict:
    """Load splits written by save_splits."""
    with open(splits_path, "rb") as f:
        return pickle.load(f)


def make_pipeline(candidate_name: str) -> Pipeline:
    """
    Build a preprocessing and model pipeline, so the fitted scaler is saved with the model.

    Continuous columns are standardized once; the binary column passes through unchanged.
    """
    if candidate_name not in CANDIDATE_FACTORY_DICT:
        raise ValueError(
            f"unknown candidate {candidate_name!r}; choose from {sorted(CANDIDATE_FACTORY_DICT)}"
        )
    preprocessor = ColumnTransformer(
        [("scale", StandardScaler(), config.CONTINUOUS_COLUMNS)], remainder="passthrough"
    )
    return Pipeline([("preprocess", preprocessor),
                     ("model", CANDIDATE_FACTORY_DICT[candidate_name]())])


def evaluate(model, X, y) -> dict:
    """Return accuracy, ROC AUC, and F1 for a fitted classifier."""
    probability = model.predict_proba(X)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "auc": float(roc_auc_score(y, probability)),
        "f1": float(f1_score(y, prediction)),
    }


def train_candidate(candidate_name: str, splits_path, output_dir: Path) -> dict:
    """
    Train one candidate on the training set and score it on the validation set.

    Parameters
    ----------
    candidate_name : str
        Key of CANDIDATE_FACTORY_DICT.
    splits_path : str or pathlib.Path
        Output of save_splits.
    output_dir : pathlib.Path
        Folder where the fitted pipeline is saved.

    Returns
    -------
    result_dict : dict
        candidate, model_path, and validation (metric dict).
    """
    split_dict = load_splits(splits_path)
    X_train, y_train = split_dict["train"]
    X_val, y_val = split_dict["validation"]

    model = make_pipeline(candidate_name).fit(X_train, y_train)
    model_path = Path(output_dir) / f"{candidate_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return {
        "candidate": candidate_name,
        "model_path": str(model_path),
        "validation": evaluate(model, X_val, y_val),
    }


def select_best(result_list: list) -> dict:
    """Return the result with the highest validation AUC, breaking ties by accuracy, then name."""
    if not result_list:
        raise ValueError("no candidate results to choose from")
    return max(
        result_list,
        key=lambda r: (r["validation"]["auc"], r["validation"]["accuracy"], r["candidate"]),
    )


def score_on_test(model_path, splits_path) -> dict:
    """Load a saved pipeline and score it on the test set."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    X_test, y_test = load_splits(splits_path)["test"]
    return evaluate(model, X_test, y_test)
