import json
from pathlib import Path

from backend.api import create_app
from backend.config import Settings
from backend.schemas import EvalCard


def test_openapi_matches_published_contract(tmp_path):
    app = create_app(
        Settings(data_dir=tmp_path, frontend_dist=tmp_path / "no-frontend", _env_file=None)
    )
    published = Path(__file__).resolve().parents[2] / "docs/api/openapi.json"
    assert app.openapi() == json.loads(published.read_text())


def test_published_eval_cards_match_workspace_response():
    folder = Path(__file__).resolve().parents[2] / "docs/results/s1"
    cards = sorted(folder.glob("card-*.json"))
    assert cards, "Published evaluation cards must be present."
    for path in cards:
        value = json.loads(path.read_text())
        assert EvalCard.model_validate(value).model_dump(exclude_unset=True) == value, path.name
