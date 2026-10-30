from backend.panel_votes import PANEL, choices, majority, overturned, parse, request_for, tally


def test_two_of_three_changes_a_label_and_ties_do_not():
    assert overturned(["b", "b", "a"], "a", "b")
    assert not overturned(["b", "a", "Both fit equally"], "a", "b")
    assert not overturned(["b", None, None], "a", "b")


def test_parse_accepts_only_a_listed_option():
    options = ["pause", "keep", "insufficient information"]
    assert parse(' "Pause". ', options) == "pause"
    assert parse("pause it", options) is None
    assert parse(None, options) is None


def test_request_never_leaks_key_fields_and_shuffle_is_per_model():
    row = dict(id="c1", message="hi", option_a="x", option_b="y", gold="x", kind="control")
    prompts = [request_for(row, m, "mtop")[0] for m in PANEL]
    assert not any("control" in p or "gold" in p for p in prompts)
    assert sorted(request_for(row, PANEL[0], "mtop")[1]) == sorted(choices(row))


def test_tally_flags_bias_above_two_percent_of_controls():
    rows = [dict(id=str(i)) for i in range(101)]
    key = {
        str(i): dict(gold="a", candidate="b", kind="control" if i else "disagreement")
        for i in range(101)
    }
    votes = {(m, "0"): "b" for m in PANEL} | {(m, "1"): "b" for m in PANEL}
    report = tally(rows, votes, key)
    assert [c["id"] for c in report["changed"]] == ["0", "1"]
    assert report["controls_overturned"] == 1 and not report["biased"]
    votes |= {(m, "2"): "b" for m in PANEL}
    assert not tally(rows, votes, key)["biased"]  # exactly 2% is allowed
    votes |= {(m, "3"): "b" for m in PANEL}
    assert tally(rows, votes, key)["biased"]


def test_majority_needs_two_matching_votes():
    assert majority(["keep", "keep", None]) == "keep"
    assert majority(["keep", "pause", None]) is None
    assert majority([None, None, None]) is None
