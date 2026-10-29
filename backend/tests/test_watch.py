import io

import pandas as pd
import pytest

from backend import watch


def row(**kwargs):
    return dict(
        round=1,
        labels=300,
        model_version="ren-1.1",
        coverage=0.4,
        precision=0.995,
        accuracy=0.9,
        promoted=True,
        guarded=False,
        **kwargs,
    )


def test_round_metrics_answers_and_guard_from_fake_events():
    output = io.StringIO()
    view = watch.LiveView(output)
    view({"kind": "round", "row": row()})
    for i in range(9):
        view(
            {
                "kind": "answer",
                "item": pd.DataFrame([dict(text=f"item {i}", label="Food")]),
                "decision": {"label": "Travel", "spot_check": i == 8},
            }
        )
    view({"kind": "round", "row": {**row(), "guarded": True}})
    rendered = output.getvalue()
    assert "Coverage [████████░░░░░░░░░░░░] 40.00%" in rendered
    assert "99.50% | meets 99% target" in rendered
    assert "Accuracy: 90.00% | Labels: 300" in rendered
    assert "guarded: true" in rendered and "all new items go to a person" in rendered
    assert "spot-check: item 8 -> Food (predicted Travel)" in rendered
    assert "item 0 ->" not in rendered
    assert "\x1b" not in rendered


def test_refusal_reasons_observe_retrain_return_and_restore_method():
    class FakeLoop:
        guarded = False
        y = ["a"] * 320

        def retrain(self):
            return {"promoted": False, "reasons": ["certified coverage drops 0.020"]}

    loop = FakeLoop()
    original = loop.retrain
    output = io.StringIO()
    view = watch.LiveView(output)
    view({"kind": "round", "row": row(), "loop": loop})
    result = loop.retrain()
    assert result == original()
    view({"kind": "round", "row": {**row(), "promoted": False, "precision": 0.98}, "loop": loop})
    view.close()
    assert loop.retrain == original
    assert "Refused: certified coverage drops 0.020" in output.getvalue()
    assert "98.00% | BELOW 99% target" in output.getvalue()
    assert "Labels: 320" in output.getvalue()


def test_cli_uses_simulation_callback_and_restores_on_failure(monkeypatch, capsys):
    calls = []

    def fake_simulate(dataset, rounds, batch, seed, strategy, **kwargs):
        calls.append((dataset, rounds, batch, seed, strategy, kwargs["featurizer"]))
        kwargs["on_step"]({"kind": "round", "row": {**row(), "precision": None}})

    monkeypatch.setattr(watch.sim, "simulate", fake_simulate)
    watch.main(["--dataset", "mtop", "--rounds", "2", "--batch", "5", "--featurizer", "char_tfidf"])
    assert calls == [("mtop", 2, 5, 0, "random", "char_tfidf")]
    assert "unmeasured | not certified" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        watch.main(["--rounds", "0"])


def test_terminal_controls_in_answer_text_are_removed():
    assert "\x1b" not in watch.text("untrusted\x1b[2J\ntext")
