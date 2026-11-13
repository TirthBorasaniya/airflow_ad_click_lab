"""
Tests for the promotion rule, the file-based registry, and notifications.
"""

import json
import math
from datetime import datetime, timezone

import pytest

from src import config, notifications, registry


@pytest.mark.parametrize(
    "candidate, champion, expected",
    [
        (0.95, None, True),
        (0.85, None, False),
        (0.960, 0.950, True),
        (0.952, 0.950, True),
        (0.951, 0.950, False),
        (0.940, 0.950, False),
        (math.nan, None, False),
    ],
)
def test_decide_promotion(candidate, champion, expected):
    o_promote, reason = registry.decide_promotion(candidate, champion, 0.9, 0.002)
    assert o_promote is expected
    assert reason


def test_promote_writes_champion_and_copies_model(artifact_dir):
    model_path = artifact_dir / "candidate.pkl"
    model_path.write_bytes(b"model")
    best_dict = {"candidate": "random_forest", "model_path": str(model_path),
                 "validation": {"auc": 0.97}, "test": {"auc": 0.96}}
    registry_path = config.registry_dir()
    assert registry.load_champion(registry_path) is None

    when = datetime(2026, 11, 1, tzinfo=timezone.utc)
    champion_dict = registry.promote(best_dict, registry_path, "manual__2026-11-01T00:00:00+00:00",
                                     now=when)
    assert registry.load_champion(registry_path) == champion_dict
    assert champion_dict["promoted_at"] == when.isoformat()
    assert open(champion_dict["model_path"], "rb").read() == b"model"
    assert ":" not in champion_dict["model_path"].split("models/")[1]


def test_history_appends_lines(artifact_dir):
    registry_path = config.registry_dir()
    registry.append_history(registry_path, {"decision": "promoted"})
    registry.append_history(registry_path, {"decision": "kept_champion"})
    line_list = (registry_path / registry.HISTORY_FILE).read_text().splitlines()
    assert [json.loads(line)["decision"] for line in line_list] == ["promoted", "kept_champion"]


def test_run_dir_is_safe_and_created(artifact_dir):
    path = config.run_dir("scheduled__2026-11-01T00:00:00+00:00")
    assert path.exists()
    assert ":" not in path.name and "+" not in path.name


def clear_smtp(monkeypatch):
    for key in notifications.SMTP_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_notification_logs_without_smtp(artifact_dir, monkeypatch):
    clear_smtp(monkeypatch)
    assert notifications.send_notification("subject", "body") == "log"
    assert "subject | body" in (artifact_dir / notifications.NOTIFICATION_LOG).read_text()


class FakeSMTP:
    """Stand-in for smtplib.SMTP that records the sent message."""
    sent_list = []

    def __init__(self, host, port, timeout):
        self.host = host

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def starttls(self):
        pass

    def login(self, user, password):
        pass

    def send_message(self, message):
        FakeSMTP.sent_list.append(message)


def test_notification_emails_with_smtp(artifact_dir, monkeypatch):
    for key, value in {"SMTP_HOST": "smtp.example.com", "SMTP_USER": "me@example.com",
                       "SMTP_PASSWORD": "secret", "NOTIFY_EMAIL_TO": "team@example.com"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(notifications.smtplib, "SMTP", FakeSMTP)
    assert notifications.send_notification("Airflow pipeline failed", "details") == "email"
    assert FakeSMTP.sent_list[-1]["To"] == "team@example.com"


def test_email_failure_falls_back_to_log(artifact_dir, monkeypatch):
    for key in notifications.SMTP_ENV_KEYS:
        monkeypatch.setenv(key, "x")

    def broken_smtp(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(notifications.smtplib, "SMTP", broken_smtp)
    assert notifications.send_notification("subject", "body") == "log"


def test_describe_context_handles_task_and_dag_contexts():
    class TI:
        dag_id, run_id, task_id, try_number = "ad_click_pipeline", "run1", "split_data", 3

    text = notifications.describe_context({"ti": TI(), "exception": ValueError("boom")})
    assert "task=split_data try=3" in text and "error=boom" in text
    assert "dag=unknown" in notifications.describe_context({})
