"""
Tests for splitting, candidate training, selection, and test scoring.
"""

import pickle

import pandas as pd
import pytest

from src import config
from src.data import load_dataset
from src.training import (
    evaluate,
    load_splits,
    make_pipeline,
    save_splits,
    score_on_test,
    select_best,
    split_dataset,
    train_candidate,
)


@pytest.fixture(scope="module")
def split_dict():
    return split_dataset(load_dataset())


def test_split_sizes_and_stratification(split_dict):
    size_list = [len(split_dict[name][0]) for name in ("train", "validation", "test")]
    assert size_list == [600, 200, 200]
    for _, y in split_dict.values():
        assert y.mean() == pytest.approx(0.5, abs=0.02)
    index_set_list = [set(X.index) for X, _ in split_dict.values()]
    assert not (index_set_list[0] & index_set_list[1] or index_set_list[0] & index_set_list[2]
                or index_set_list[1] & index_set_list[2])


def test_unknown_candidate_is_rejected():
    with pytest.raises(ValueError, match="unknown candidate"):
        make_pipeline("svm")


@pytest.mark.parametrize("name", config.CANDIDATE_NAMES)
def test_every_candidate_trains_and_saves(split_dict, tmp_path, name):
    splits_path = save_splits(split_dict, tmp_path / "splits.pkl")
    result_dict = train_candidate(name, splits_path, tmp_path)
    assert result_dict["candidate"] == name
    assert set(result_dict["validation"]) == {"accuracy", "auc", "f1"}
    assert result_dict["validation"]["auc"] > 0.9

    # the saved pipeline scores raw rows directly because it includes the scaler
    with open(result_dict["model_path"], "rb") as f:
        model = pickle.load(f)
    raw_rows = split_dict["test"][0].head(3)
    assert model.predict_proba(raw_rows).shape == (3, 2)


def test_score_on_test_matches_direct_evaluation(split_dict, tmp_path):
    splits_path = save_splits(split_dict, tmp_path / "splits.pkl")
    result_dict = train_candidate("logistic_regression", splits_path, tmp_path)
    with open(result_dict["model_path"], "rb") as f:
        model = pickle.load(f)
    X_test, y_test = load_splits(splits_path)["test"]
    assert score_on_test(result_dict["model_path"], splits_path) == evaluate(model, X_test, y_test)


def make_result(name, auc, accuracy=0.9):
    return {"candidate": name, "model_path": f"/{name}.pkl",
            "validation": {"auc": auc, "accuracy": accuracy, "f1": 0.9}}


def test_select_best_picks_highest_auc_then_accuracy():
    result_list = [make_result("a", 0.95), make_result("b", 0.97), make_result("c", 0.96)]
    assert select_best(result_list)["candidate"] == "b"
    tie_list = [make_result("a", 0.97, 0.90), make_result("b", 0.97, 0.92)]
    assert select_best(tie_list)["candidate"] == "b"


def test_select_best_rejects_empty_list():
    with pytest.raises(ValueError):
        select_best([])


def test_scaling_is_applied_once():
    model = make_pipeline("logistic_regression")
    X = pd.DataFrame([[60.0, 30, 50000.0, 180.0, 1], [70.0, 40, 60000.0, 200.0, 0]],
                     columns=config.FEATURE_COLUMNS)
    transformed = model.named_steps["preprocess"].fit_transform(X)
    assert transformed.shape == (2, len(config.FEATURE_COLUMNS))
