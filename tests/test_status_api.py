"""
Tests for the Flask status service with the Airflow API replaced by fakes.
"""

import json

import pytest

import app as status_app


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(status_app, "ARTIFACT_DIR", tmp_path)
    status_app.token_cache_dict.update({"token": None, "expires_at": 0.0})
    return status_app.app.test_client()


def fake_airflow(monkeypatch, state, run_list=None):
    """Replace HTTP calls with an Airflow API whose latest run has the given state."""
    if run_list is None:
        run_list = [{"state": state, "dag_run_id": "manual__1", "run_after": "2026-11-01",
                     "start_date": "2026-11-01T00:00:00", "end_date": "2026-11-01T00:01:00"}]
    monkeypatch.setattr(status_app.requests, "post",
                        lambda *a, **k: FakeResponse(201, {"access_token": "token"}))
    monkeypatch.setattr(status_app.requests, "get",
                        lambda *a, **k: FakeResponse(200, {"dag_runs": run_list}))


def write_artifacts(artifact_dir):
    (artifact_dir / "registry").mkdir()
    champion_dict = {"run_id": "manual__1", "candidate": "random_forest",
                     "test": {"auc": 0.97, "accuracy": 0.95}}
    (artifact_dir / "registry" / "champion.json").write_text(json.dumps(champion_dict))
    summary_dict = {"run_id": "manual__1",
                    "decision": {"decision": "promoted", "reason": "no current champion"}}
    (artifact_dir / "latest_summary.json").write_text(json.dumps(summary_dict))
    scoring_dict = {"rows_scored": 200, "predicted_click_rate": 0.48,
                    "champion_candidate": "random_forest"}
    (artifact_dir / "latest_scoring.json").write_text(json.dumps(scoring_dict))


def test_health(client):
    assert client.get("/health").data == b"ok"


@pytest.mark.parametrize("state, target", [("success", "/success"), ("failed", "/failure"),
                                           ("running", "/failure")])
def test_index_redirects_by_state(client, monkeypatch, state, target):
    fake_airflow(monkeypatch, state)
    response = client.get("/")
    assert response.status_code == 302
    assert response.headers["Location"].endswith(target)


def test_success_page_shows_champion_and_decision(client, monkeypatch, tmp_path):
    fake_airflow(monkeypatch, "success")
    write_artifacts(tmp_path)
    page = client.get("/success").get_data(as_text=True)
    assert "random_forest" in page and "0.9700" in page and "promoted" in page
    assert "48.0%" in page and "200" in page


def test_failure_page_without_artifacts(client, monkeypatch):
    fake_airflow(monkeypatch, "failed")
    page = client.get("/failure").get_data(as_text=True)
    assert "Pipeline Not Successful" in page and "FAILED" in page


def test_running_shows_in_progress(client, monkeypatch):
    fake_airflow(monkeypatch, "running")
    page = client.get("/failure").get_data(as_text=True)
    assert "Pipeline In Progress" in page


def test_api_status_json(client, monkeypatch, tmp_path):
    fake_airflow(monkeypatch, "success")
    write_artifacts(tmp_path)
    body = client.get("/api/status").get_json()
    assert body["latest_run"]["state"] == "success"
    assert body["champion"]["candidate"] == "random_forest"
    assert body["scoring"]["rows_scored"] == 200


def test_no_runs_yet(client, monkeypatch):
    fake_airflow(monkeypatch, None, run_list=[])
    body = client.get("/api/status").get_json()
    assert "no runs yet" in body["latest_run"]["note"]


def test_authentication_failure_is_reported(client, monkeypatch):
    monkeypatch.setattr(status_app.requests, "post",
                        lambda *a, **k: FakeResponse(401, {"detail": "bad credentials"}))
    body = client.get("/api/status").get_json()
    assert "could not authenticate" in body["latest_run"]["note"]


def test_network_error_is_reported(client, monkeypatch):
    def raise_error(*args, **kwargs):
        raise status_app.requests.ConnectionError("airflow down")

    monkeypatch.setattr(status_app.requests, "post", raise_error)
    body = client.get("/api/status").get_json()
    assert "could not authenticate" in body["latest_run"]["note"]


def test_token_is_cached(client, monkeypatch):
    call_list = []

    def counting_post(*args, **kwargs):
        call_list.append(1)
        return FakeResponse(201, {"access_token": "token"})

    monkeypatch.setattr(status_app.requests, "post", counting_post)
    monkeypatch.setattr(status_app.requests, "get",
                        lambda *a, **k: FakeResponse(200, {"dag_runs": []}))
    client.get("/api/status")
    client.get("/api/status")
    assert len(call_list) == 1
