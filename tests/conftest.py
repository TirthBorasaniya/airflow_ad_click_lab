"""
Shared test setup: make the DAG package and status service importable, and isolate artifacts.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dags"))
sys.path.insert(0, str(REPO_ROOT / "status_api"))


@pytest.fixture
def artifact_dir(tmp_path, monkeypatch):
    """Point every artifact path at a temporary folder."""
    from src import config
    monkeypatch.setattr(config, "ARTIFACT_DIR", tmp_path)
    return tmp_path
