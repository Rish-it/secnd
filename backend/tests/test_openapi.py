import json
from pathlib import Path

from backend.api import create_app
from backend.config import Settings


def test_openapi_matches_published_contract(tmp_path):
    app = create_app(
        Settings(data_dir=tmp_path, frontend_dist=tmp_path / "no-frontend", _env_file=None)
    )
    published = Path(__file__).resolve().parents[2] / "docs/api/openapi.json"
    assert app.openapi() == json.loads(published.read_text())
