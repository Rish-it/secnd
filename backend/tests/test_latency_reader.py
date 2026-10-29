import pytest

from backend.latency_reader import statistics


def test_latency_reports_tail_and_sequential_throughput():
    result = statistics([10] * 90 + [2000] * 10)
    assert result["p50_ms"] == 10
    assert result["p95_ms"] == 2000
    assert result["throughput_requests_per_second"] == pytest.approx(100_000 / 20_900)
    assert result["requests"] == 100
