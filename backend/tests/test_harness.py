from types import SimpleNamespace

from backend.config import Settings
from backend.harness import Harness
from backend.meta_webhook import Message
from backend.store import Store


def test_workspace_failure_does_not_stop_others_and_kill_persists(tmp_path):
    import threading

    store = Store(tmp_path / "db")
    calls = []

    def sync(ws, body):
        calls.append(ws)
        if ws == "broken":
            raise RuntimeError("provider-private-value")
        return []

    service = SimpleNamespace(store=store, _lock=lambda ws: threading.RLock())
    ads = SimpleNamespace(status=lambda ws: SimpleNamespace(sandbox_allowed=True), sync=sync)
    h = Harness(Settings(_env_file=None), service, ads, None)
    h.outcomes.collect = lambda ws, today: None
    for ws in ["broken", "working"]:
        h.configure(ws, enabled=True)
    h.tick()
    assert calls == ["broken", "working"]
    assert h.status("broken")["activity"][0]["event"] == "cycle_failed"
    assert h.status("working")["activity"][0]["event"] == "cycle_completed"
    with store.connect(write=True) as db:
        db.execute("UPDATE agent_control SET killed=1")
    calls.clear()
    h.tick()
    assert calls == []
    assert Harness(Settings(_env_file=None), service, ads, None).killed()


def test_webhook_queue_is_durable_paused_and_idempotent(tmp_path):
    import threading

    store = Store(tmp_path / "db")
    service = SimpleNamespace(store=store, _lock=lambda ws: threading.RLock())
    seen = []
    webhook = SimpleNamespace(
        handle=lambda ws, message: seen.append(message.item_id) or {"status": "queued"}
    )
    ads = SimpleNamespace(status=lambda ws: SimpleNamespace(sandbox_allowed=False))
    h = Harness(Settings(_env_file=None), service, ads, webhook)
    m = Message(platform="whatsapp", message_id="one", sender="1", account="2", text="synthetic")
    assert not h.enqueue("ws", m)["duplicate"]
    assert h.enqueue("ws", m)["duplicate"]
    h.tick()
    assert seen == []
    h = Harness(Settings(_env_file=None), service, ads, webhook)
    h.configure("ws", enabled=True)
    h.tick()
    h.tick()
    assert seen == [m.item_id]
    h.close()
