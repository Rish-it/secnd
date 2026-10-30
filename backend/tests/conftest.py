"""Default test fixtures explicitly use the small head path; reader tests opt in with fakes."""

import pytest


@pytest.fixture(autouse=True)
def test_serving_backend(monkeypatch):
    # No production checkpoint downloads, GPU allocation or live provider calls in tests.
    monkeypatch.setenv("SECND_SERVING_BACKEND", "head")
