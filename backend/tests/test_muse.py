import json

import httpx
import pytest

from backend.config import Settings
from backend.muse import MuseError, smoke


def settings():
    return Settings(muse_api_key="test-only-key", _env_file=None)


def test_smoke_sends_one_bounded_request_and_returns_only_metadata():
    requests = []

    def handler(request):
        requests.append(request)
        body = json.loads(request.content)
        assert request.headers["Authorization"] == "Bearer test-only-key"
        assert body["model"] == "meta-models/muse-spark-open-1026"
        assert body["max_tokens"] == 64
        assert "response_format" not in body
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"message": {"content": "ready"}}],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 10,
                    "completion_tokens_details": {"reasoning_tokens": 8},
                },
            },
        )

    result = smoke(settings(), transport=httpx.MockTransport(handler))
    assert len(requests) == 1
    assert result.usage.reasoning_tokens == 8
    assert "ready" not in result.model_dump_json()
    assert "test-only-key" not in result.model_dump_json()


def test_smoke_sanitizes_provider_errors_and_requires_a_key(monkeypatch):
    transport = httpx.MockTransport(lambda request: httpx.Response(401, text="sensitive response"))
    with pytest.raises(MuseError, match="HTTP 401") as caught:
        smoke(settings(), transport=transport)
    assert "sensitive" not in str(caught.value)
    monkeypatch.delenv("SECND_MUSE_API_KEY", raising=False)
    with pytest.raises(MuseError, match="SECND_MUSE_API_KEY"):
        smoke(Settings(muse_api_key="", _env_file=None), transport=transport)
