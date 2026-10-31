from datetime import date

from backend.tests.test_meta_ads import ads, sync  # noqa: F401


def test_outcome_waits_seven_days_and_exports_review_and_metrics(ads):  # noqa: F811
    client, service, settings, state, transport = ads
    h = service.harness
    h.ads = client.app.state.meta_ads
    proposal = sync(client).json()[0]
    with service.store.connect(write=True) as db:
        db.execute("UPDATE ad_outcomes SET due_date='2026-10-08'")
    h.outcomes.collect("ads", today=date(2026, 10, 7))
    assert h.outcomes.rows("ads")[0]["after"] is None
    state["spend"] = "55.00"
    # Match the fixed fixture's completed seven-day observation window.
    from backend.tests import test_meta_ads

    old = test_meta_ads.DATES.copy()
    try:
        test_meta_ads.DATES.update(since="2026-10-01", until="2026-10-07")
        h.outcomes.collect("ads", today=date(2026, 10, 8))
        row = client.get("/api/v1/workspaces/ads/outcomes").json()[0]
        assert row["before"]["spend"] == "42.00"
        assert row["after"]["spend"] == "55.00"
        assert row["after"]["ctr"] == "1.75"
        calls = len(state["calls"])
        h.outcomes.collect("ads", today=date(2026, 10, 9))
        assert len(state["calls"]) == calls
        exported = client.get("/api/v1/workspaces/ads/outcomes.csv")
        assert exported.status_code == 200 and "55.00" in exported.text
        assert proposal["item_id"] in exported.text
        assert "ads-test-secret" not in exported.text
    finally:
        test_meta_ads.DATES.update(old)


def test_observed_outcome_requires_review_then_uses_existing_learning_and_rollback(ads):  # noqa: F811
    client, service, settings, state, _ = ads
    before = {"spend": "42", "cpa": "7", "ctr": "1.75", "currency": "USD"}
    after = {"spend": "55", "cpa": "8", "ctr": "1.50", "currency": "USD"}
    holdout = service.store.workspace("ads")["calib"]
    original = service.card("ads").model_version
    review = service.queue_outcome_review("ads", "decision", before, after)
    assert service.queue_outcome_review("ads", "decision", before, after) == review
    assert not service.store.latest_prediction("ads", review)["auto"]
    assert all(r["item_id"] != review for r in service.store.labels("ads"))
    response = client.post(
        "/api/v1/workspaces/ads/answers",
        json={"item_id": review, "label": "lower budget"},
        headers={"Idempotency-Key": "outcome-review"},
    )
    assert response.status_code == 200
    service.drain()
    assert service.store.labels("ads")[-1]["label"] == "lower budget"
    assert service.autonomy.status("ads", "ad")["reviewed"] == 0
    assert service.store.workspace("ads")["calib"] == holdout
    assert client.post(f"/api/v1/workspaces/ads/models/{original}:rollback").status_code == 200
