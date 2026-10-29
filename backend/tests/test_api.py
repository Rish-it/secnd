from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api import create_app
from backend.config import Settings
from backend.data import MAX_CSV_BYTES
from backend.schemas import (
    AnswerResult,
    ChatResult,
    Decision,
    Error,
    MetricsResult,
    Page,
)


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings(
        data_dir=tmp_path / "storage",
        frontend_dist=tmp_path / "missing-dist",
        _env_file=None,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_health_checks_storage_and_reports_failure(client: TestClient, tmp_path: Path):
    assert client.get("/api/health").json() == {"status": "ok", "database": "ready"}
    (tmp_path / "storage" / "secnd.sqlite3").unlink()
    assert client.get("/api/health").status_code == 503


def test_health_rejects_corrupt_storage(client: TestClient, tmp_path: Path):
    (tmp_path / "storage" / "secnd.sqlite3").write_bytes(b"not a SQLite database")
    assert client.get("/api/health").status_code == 503


def test_csv_preview_handles_bom_quotes_and_blank_rows_without_saving(client: TestClient):
    response = client.post(
        "/api/preview",
        files={
            "file": (
                "decisions.csv",
                '\ufeffdescription,amount,category\n"Train, return",25,Travel\n\n'.encode(),
                "text/csv",
            )
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "filename": "decisions.csv",
        "columns": ["description", "amount", "category"],
        "row_count": 1,
        "rows": [{"description": "Train, return", "amount": "25", "category": "Travel"}],
    }


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"description,category\n",
        b"description, description\nx,y\n",
        b"description,category\nx,y,z\n",
        b"description,category\n\xff,Travel\n",
        b"description,category\n\x00,Travel\n",
        b'description,category\n"unfinished,Travel\n',
        b"description\nTrain\n",
    ],
)
def test_invalid_csv_has_a_useful_error(client: TestClient, content: bytes):
    response = client.post("/api/preview", files={"file": ("bad.csv", content, "text/csv")})
    assert response.status_code == 422
    assert response.json()["detail"]


def test_preview_limits_sample_and_counts_all_rows(client: TestClient):
    content = "description,category\n" + "Train,Travel\n" * 12
    response = client.post("/api/preview", files={"file": ("data.csv", content, "text/csv")})
    assert response.status_code == 200
    assert response.json()["row_count"] == 12
    assert len(response.json()["rows"]) == 5


def test_upload_size_is_bounded(client: TestClient):
    response = client.post(
        "/api/preview", files={"file": ("big.csv", b"x" * (MAX_CSV_BYTES + 1), "text/csv")}
    )
    assert response.status_code == 413


def test_api_misses_do_not_serve_frontend_html(tmp_path: Path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>Secnd</html>")
    (dist / "assets" / "app.js").write_text("console.log('Secnd')")
    with TestClient(
        create_app(Settings(data_dir=tmp_path / "storage", frontend_dist=dist, _env_file=None))
    ) as client:
        assert client.get("/").text == "<html>Secnd</html>"
        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/api/missing").status_code == 404
        assert client.get("/.env").status_code == 404


API = "/api/v1/workspaces/default"


def test_invalid_single_label_import_and_unprepared_prediction_fixtures(client: TestClient):
    response = client.post(
        f"{API}/decisions:import",
        files={"file": ("data.csv", b"text,category\nTrain,Travel\n", "text/csv")},
        data={"target_column": "category"},
    )
    assert response.status_code == 422
    response = client.post(
        f"{API}/predict",
        json={"items": [{"item_id": "a", "text": "Train", "date": "2026-10-29"}]},
    )
    assert response.status_code == 200
    predicted = Decision.model_validate(response.json()[0])
    assert predicted.item_id == "a"
    assert predicted.item.date.isoformat() == "2026-10-29"
    assert not predicted.auto
    assert predicted.threshold is None


@pytest.mark.parametrize(
    ("route", "schema"),
    [("feed", Page[Decision]), ("queue", Page[Decision])],
)
def test_contract_pages(client: TestClient, route, schema):
    response = client.get(f"{API}/{route}")
    assert response.headers["X-Secnd-Mode"] == "fixture"
    result = schema.model_validate(response.json())
    assert len(result.items) == 1
    assert result.next_cursor is None


def test_contract_answers_metrics_rollback_and_chat(client: TestClient):
    response = client.post(
        f"{API}/answers",
        json={"item_id": "a", "label": "Travel"},
        headers={"Idempotency-Key": "answer-a"},
    )
    assert AnswerResult.model_validate(response.json()).retrain == "fixture"
    card = MetricsResult.model_validate(client.get(f"{API}/metrics").json()).card
    assert card.coverage == 0.3168
    assert card.precision == 0.9962
    response = client.post(f"{API}/models/fixture-v0:rollback")
    assert response.status_code == 404
    response = client.post(f"{API}/chat", json={"question": "How many today?"})
    assert ChatResult.model_validate(response.json()).model_version is None


def test_contract_errors_and_boundaries(client: TestClient):
    for response in [
        client.post(f"{API}/predict", json={"items": []}),
        client.post(
            f"{API}/predict",
            json={"items": [{"item_id": str(i), "text": "Train"} for i in range(257)]},
        ),
        client.post(f"{API}/answers", json={"item_id": "a", "label": "Travel"}),
        client.get(f"{API}/queue?limit=0"),
        client.get(f"{API}/feed?cursor=unknown"),
        client.get("/api/v1/workspaces/bad.workspace/metrics"),
        client.post(
            f"{API}/decisions:import",
            files={"file": ("data.csv", b"text,category\nTrain,Travel\n", "text/csv")},
            data={"target_column": "missing"},
        ),
    ]:
        assert response.status_code == 422
        assert Error.model_validate(response.json()).message
    response = client.post(f"{API}/models/missing:rollback")
    assert response.status_code == 404
    assert Error.model_validate(response.json()).code == "http_404"
    assert client.get(f"{API}/queue?cursor=fixture-end").json()["items"] == []


def test_real_evaluations_are_distinct_from_fixture_workspace_data(client):
    for dataset in ("mtop", "banking77", "ipsa"):
        response = client.get(f"{API}/metrics?dataset={dataset}")
        assert response.status_code == 200
        assert response.headers["X-Secnd-Mode"] == "evaluation"
        measured = MetricsResult.model_validate(response.json())
        assert measured.dataset == dataset
        assert measured.card.model_version.startswith(dataset)
        assert measured.history == []
        assert measured.curves.reliability and measured.curves.coverage_precision
    assert client.get(f"{API}/metrics?dataset=unknown").status_code == 422
    assert client.get(f"{API}/models").json() == {"items": [], "next_cursor": None}


def test_registry_models_promote_and_rollback_are_workspace_scoped(client, tmp_path):
    from backend.registry import Registry

    registry = Registry(tmp_path / "storage", "default")
    first = registry.save({}, {"coverage": 0.1})
    second = registry.save({}, {"coverage": 0.2})
    registry.promote(first)
    versions = client.get(f"{API}/models")
    assert versions.headers["X-Secnd-Mode"] == "live"
    assert [v["current"] for v in versions.json()["items"]] == [True, False]
    assert client.post(f"{API}/models/{second}:promote").json()["current"]
    assert registry.current() == second
    assert client.post(f"{API}/models/{first}:rollback").json()["card"]["coverage"] == 0.1
    assert registry.current() == first
    assert client.get("/api/v1/workspaces/other/models").json()["items"] == []
    assert client.post(f"{API}/models/missing:promote").status_code == 404
    incomplete = registry.save({}, {"threshold": None})
    assert client.post(f"{API}/models/{incomplete}:promote").status_code == 503
    assert registry.current() == first
    assert client.get(f"{API}/models").status_code == 503


def test_missing_published_evaluation_returns_typed_404(tmp_path):
    settings = Settings(data_dir=tmp_path, evaluation_dir=tmp_path / "absent", _env_file=None)
    with TestClient(create_app(settings)) as client:
        response = client.get(f"{API}/metrics")
        assert response.status_code == 404
        assert Error.model_validate(response.json()).code == "http_404"
