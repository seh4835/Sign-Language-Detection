import numpy as np
from fastapi.testclient import TestClient

import src.api.app as api


client = TestClient(api.app)


def setup_function():
    api._reset_artifact_cache()


def teardown_function():
    api._reset_artifact_cache()


def valid_payload():
    return {"sequence": np.zeros((30, 180), dtype=np.float32).tolist()}


def test_health_endpoint_reports_loaded_artifacts(monkeypatch):
    monkeypatch.setattr(api, "load_artifact", lambda _path: (object(), object(), ["bad"]))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_valid_prediction_uses_existing_prediction_function(monkeypatch):
    monkeypatch.setattr(api, "load_artifact", lambda _path: (object(), object(), ["bad", "good"]))
    calls = {}

    def fake_predict(classifier, normalizer, sequence):
        calls["shape"] = sequence.shape
        return 1, 0.875

    monkeypatch.setattr(api, "predict_sequence", fake_predict)

    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200
    assert response.json() == {
        "predicted_sign": "good",
        "confidence": 0.875,
        "status": "ok",
    }
    assert calls["shape"] == (30, 180)


def test_prediction_rejects_incorrect_frame_count(monkeypatch):
    monkeypatch.setattr(api, "load_artifact", lambda _path: (object(), object(), ["bad"]))

    response = client.post(
        "/predict",
        json={"sequence": np.zeros((29, 180), dtype=np.float32).tolist()},
    )

    assert response.status_code == 422
    assert "exactly 30 frames" in response.text


def test_prediction_rejects_incorrect_feature_dimension(monkeypatch):
    monkeypatch.setattr(api, "load_artifact", lambda _path: (object(), object(), ["bad"]))

    response = client.post(
        "/predict",
        json={"sequence": np.zeros((30, 179), dtype=np.float32).tolist()},
    )

    assert response.status_code == 422
    assert "exactly 180 features" in response.text


def test_prediction_rejects_invalid_numeric_values(monkeypatch):
    monkeypatch.setattr(api, "load_artifact", lambda _path: (object(), object(), ["bad"]))
    payload = valid_payload()
    payload["sequence"][0][0] = "not-a-number"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_health_reports_unavailable_model_artifacts(monkeypatch, tmp_path):
    def missing_artifacts(_path):
        raise FileNotFoundError("classifier.pkl is missing")

    monkeypatch.setattr(api, "ARTIFACTS", tmp_path / "missing")
    monkeypatch.setattr(api, "load_artifact", missing_artifacts)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "unavailable",
        "model_loaded": False,
        "detail": "classifier.pkl is missing",
    }
