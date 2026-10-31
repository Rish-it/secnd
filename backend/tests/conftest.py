import pytest


@pytest.fixture(autouse=True)
def explicit_fixture_mode(monkeypatch):
    # Legacy contract/connector tests intentionally exercise the synthetic demo.
    monkeypatch.setenv("SECND_DEMO_MODE", "true")
    monkeypatch.setenv("SECND_FEATURIZER", "char_tfidf")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
