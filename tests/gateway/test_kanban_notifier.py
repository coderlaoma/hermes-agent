import asyncio
import re
import time
from pathlib import Path


from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli import kanban_db as kb


class RecordingAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})


class DisconnectedAdapters(dict):
    """Expose a platform during collection, then simulate disconnect on get()."""

    def get(self, key, default=None):
        return None


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _make_runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_states = {}
    return runner


def _make_kanban_slash_runner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._kanban_notifier_profile = "default"
    return runner


def _kanban_event(text, *, chat_id="chat-1", thread_id="topic-1"):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            thread_id=thread_id,
            user_id="user-1",
        ),
    )


def _created_id(output):
    match = re.search(r"\b(t_[0-9a-f]+)\b", output)
    assert match, output
    return match.group(1)


def _create_completed_subscription(summary="done once"):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="notify once", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.complete_task(conn, tid, summary=summary)
        return tid
    finally:
        conn.close()


def _create_root_only_blocked_child(reason="review-required: needs human eyes"):
    conn = kb.connect()
    try:
        root = kb.create_task(conn, title="root task", assignee="planner")
        child = kb.create_task(
            conn,
            title="child task",
            assignee="worker",
            parents=[root],
            initial_status="running",
        )
        kb.add_notify_sub(conn, task_id=root, platform="telegram", chat_id="chat-1")
        assert kb.list_notify_subs(conn, child) == []
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (child,))
        assert kb.block_task(conn, child, reason=reason)
        return root, child
    finally:
        conn.close()


def _unseen_terminal_events(tid):
    conn = kb.connect()
    try:
        _, events = kb.unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        return events
    finally:
        conn.close()


def test_kanban_notifier_dedupes_board_slugs_pointing_to_same_db(tmp_path, monkeypatch):
    db_path = tmp_path / "shared-kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    kb.write_board_metadata("alias-a", name="Alias A")
    kb.write_board_metadata("alias-b", name="Alias B")

    tid = _create_completed_subscription()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert "Kanban" in adapter.sent[0]["text"]
    assert tid in adapter.sent[0]["text"]


def test_kanban_notifier_claim_prevents_second_watcher_send(tmp_path, monkeypatch):
    db_path = tmp_path / "single-owner.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    tid = _create_completed_subscription()

    adapter1 = RecordingAdapter()
    adapter2 = RecordingAdapter()

    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter1)))
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter2)))

    assert len(adapter1.sent) == 1
    assert adapter2.sent == []


def test_kanban_notifier_rewinds_claim_if_adapter_disconnects(tmp_path, monkeypatch):
    db_path = tmp_path / "adapter-disconnect.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = DisconnectedAdapters({Platform.TELEGRAM: RecordingAdapter()})
    runner._kanban_sub_fail_states = {}

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert [ev.kind for ev in _unseen_terminal_events(tid)] == ["completed"]


def test_kanban_db_path_is_test_isolated_from_real_home():
    hermes_home = Path(kb.kanban_home())
    production_db = Path.home() / ".hermes" / "kanban.db"
    assert kb.kanban_db_path().resolve() != production_db.resolve()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
    finally:
        conn.close()

    assert kb.kanban_db_path().resolve().is_relative_to(hermes_home.resolve())
    assert kb.kanban_db_path().resolve() != production_db.resolve()


class FailingAdapter:
    """Adapter whose send() always raises, simulating a transient send error."""

    def __init__(self):
        self.attempts = 0

    async def send(self, chat_id, text, metadata=None):
        self.attempts += 1
        raise RuntimeError("simulated send failure")


class FalseResultAdapter:
    """Adapter whose send() reports failure without raising."""

    def __init__(self):
        self.attempts = 0

    async def send(self, chat_id, text, metadata=None):
        from gateway.platforms.base import SendResult

        self.attempts += 1
        return SendResult(success=False, error="simulated false result")


def test_kanban_notifier_rewinds_claim_on_send_exception(tmp_path, monkeypatch):
    """A raising adapter rewinds the claim so the next tick can retry.

    This is the second rewind path (distinct from the adapter-disconnect path
    in test_kanban_notifier_rewinds_claim_if_adapter_disconnects). Here the
    adapter is connected and the send call actually fires; the claim must
    still rewind so the event isn't lost when send() raises mid-tick.
    """
    db_path = tmp_path / "send-failure.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    adapter = FailingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # Send was attempted (so we exercised the failure path, not just the
    # disconnect path) and the claim was rewound — the unseen-events query
    # still returns the event for retry on the next tick.
    assert adapter.attempts >= 1, "send should have been attempted at least once"
    assert [ev.kind for ev in _unseen_terminal_events(tid)] == ["completed"]


def test_kanban_notifier_rewinds_claim_on_send_false_result(tmp_path, monkeypatch):
    """A SendResult(success=False) is a failed delivery, not a delivered ping.

    Some platform adapters surface API-level delivery failures as a
    SendResult instead of raising. The notifier must treat that exactly like
    an exception: keep the subscription and rewind the pre-send claim so the
    blocked/completed event is retryable instead of silently consumed.
    """
    db_path = tmp_path / "send-false-result.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    adapter = FalseResultAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.attempts >= 1, "send should have been attempted at least once"
    assert [ev.kind for ev in _unseen_terminal_events(tid)] == ["completed"]


def test_kanban_notifier_keeps_subscription_after_repeated_send_failures(tmp_path, monkeypatch):
    """Repeated failures must not silently unsubscribe a terminal notification.

    Dropping the subscription after retries leaves the human with no visible
    blocked/completed notification and no future retry path. Keep it and leave
    the event unseen so delivery recovers when the adapter/chat recovers.
    """
    db_path = tmp_path / "send-repeated-failure.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    adapter = FailingAdapter()
    runner = _make_runner(adapter)

    for _ in range(3):
        runner._running = True
        asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
    finally:
        conn.close()

    assert adapter.attempts >= 3
    assert len(subs) == 1
    assert [ev.kind for ev in _unseen_terminal_events(tid)] == ["completed"]


def test_kanban_notifier_backs_off_after_repeated_send_failures(tmp_path, monkeypatch):
    """After MAX_SEND_FAILURES, ticks inside the backoff window skip the send.

    Without backoff a permanently dead chat (bot kicked, channel deleted)
    would burn an adapter.send call and a warning log line every tick,
    forever. The backoff window suppresses the attempt entirely — no claim,
    no rewind, no log — while keeping the subscription and the unseen event
    so delivery still recovers if the chat comes back.
    """
    db_path = tmp_path / "send-backoff.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    adapter = FailingAdapter()
    runner = _make_runner(adapter)

    # Burn through the transient-failure budget (MAX_SEND_FAILURES = 3).
    for _ in range(3):
        runner._running = True
        asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert adapter.attempts == 3

    sub_key = (tid, "telegram", "chat-1", "")
    state = runner._kanban_sub_fail_states[sub_key]
    assert state["fails"] == 3
    assert state["next_retry_at"] > time.monotonic()

    # Ticks inside the backoff window must not attempt the send at all.
    for _ in range(2):
        runner._running = True
        asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert adapter.attempts == 3, "backoff window should suppress send attempts"
    assert [ev.kind for ev in _unseen_terminal_events(tid)] == ["completed"]

    # Once the window expires, retry resumes and backoff grows.
    state["next_retry_at"] = time.monotonic() - 1
    runner._running = True
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert adapter.attempts == 4
    assert runner._kanban_sub_fail_states[sub_key]["fails"] == 4


def test_kanban_notifier_recovers_after_backoff_when_chat_comes_back(tmp_path, monkeypatch):
    """A send success during a backoff retry delivers and clears the state."""
    db_path = tmp_path / "send-backoff-recovery.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    failing = FailingAdapter()
    runner = _make_runner(failing)
    for _ in range(3):
        runner._running = True
        asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    sub_key = (tid, "telegram", "chat-1", "")
    assert runner._kanban_sub_fail_states[sub_key]["fails"] == 3

    # Chat recovers: swap in a working adapter and expire the window.
    recording = RecordingAdapter()
    runner.adapters = {Platform.TELEGRAM: recording}
    runner._kanban_sub_fail_states[sub_key]["next_retry_at"] = time.monotonic() - 1

    runner._running = True
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(recording.sent) == 1
    assert tid in recording.sent[0]["text"]
    assert sub_key not in runner._kanban_sub_fail_states
    assert _unseen_terminal_events(tid) == []


def test_notifier_redelivers_same_kind_on_dispatch_cycle(tmp_path, monkeypatch):
    """A retry cycle (crashed → reclaimed → crashed) notifies the user twice.

    Before #21398 the notifier auto-unsubscribed on any terminal event kind
    (gave_up / crashed / timed_out), so the second crash in a respawn cycle
    silently dropped — the subscription was already gone. This test pins the
    new contract: subscription survives non-final terminal events; the
    cursor handles dedup.

    Two crashes ten seconds apart on the same task — both should land on
    the adapter.
    """
    db_path = tmp_path / "redeliver-cycle.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="cycle test", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        # First crash — fired by the dispatcher when the worker PID dies.
        kb._append_event(conn, tid, kind="crashed")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # First crash delivered.
    assert len(adapter.sent) == 1
    assert "crashed" in adapter.sent[0]["text"].lower()

    # Subscription survives — the cursor advanced past event #1, but the
    # row is still there.
    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1, (
            "Subscription must survive a crashed event so a respawn-cycle "
            "second crash also notifies the user (issue #21398)."
        )

        # Second crash — same task, same dispatcher (or a respawn). Append
        # another event to simulate the dispatcher firing crashed a second
        # time during retry.
        kb._append_event(conn, tid, kind="crashed")
    finally:
        conn.close()

    # New tick: the second event has a fresh id past the cursor advance,
    # so it gets claimed and delivered.
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 2, (
        f"Second crashed event should also notify; got {len(adapter.sent)} "
        f"deliveries (texts: {[d['text'] for d in adapter.sent]})"
    )
    assert "crashed" in adapter.sent[1]["text"].lower()


def test_follow_up_child_inherits_completed_parent_subscription_idempotently(tmp_path, monkeypatch):
    db_path = tmp_path / "post-completion-child-inherit.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        root = kb.create_task(conn, title="entry", assignee="planner")
        kb.add_notify_sub(
            conn,
            task_id=root,
            platform="telegram",
            chat_id="chat-1",
            thread_id="topic-1",
            user_id="user-1",
            notifier_profile="default",
        )
        kb.complete_task(conn, root, summary="root done")
        child = kb.create_task(conn, title="follow up", assignee="worker", parents=[root])
        assert kb.inherit_notify_subs_from_completed_parents(
            conn, child_task_id=child, parent_task_ids=[root]
        ) == 0
        subs = kb.list_notify_subs(conn, child)
    finally:
        conn.close()

    assert len(subs) == 1
    assert subs[0]["platform"] == "telegram"
    assert subs[0]["chat_id"] == "chat-1"
    assert subs[0]["thread_id"] == "topic-1"
    assert subs[0]["user_id"] == "user-1"
    assert subs[0]["notifier_profile"] == "default"

    conn = kb.connect()
    try:
        # Dispatcher workers complete tasks from running state; simulate that
        # so this focused notifier test exercises the actual terminal event.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (child,))
        assert kb.complete_task(conn, child, summary="follow-up done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert len(adapter.sent) == 2
    child_delivery = [item for item in adapter.sent if child in item["text"]]
    assert len(child_delivery) == 1
    assert child_delivery[0]["chat_id"] == "chat-1"
    assert child_delivery[0]["metadata"] == {"thread_id": "topic-1"}


def test_live_parent_child_does_not_inherit_subscription(tmp_path, monkeypatch):
    db_path = tmp_path / "live-parent-child-no-inherit.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        root = kb.create_task(conn, title="live root", assignee="planner")
        kb.add_notify_sub(conn, task_id=root, platform="telegram", chat_id="chat-1")
        child = kb.create_task(conn, title="ordinary child", assignee="worker", parents=[root])
        assert kb.list_notify_subs(conn, child) == []
    finally:
        conn.close()


def test_kanban_create_live_parent_child_does_not_report_subscription(tmp_path, monkeypatch):
    db_path = tmp_path / "slash-live-parent-child-no-subscribe.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        root = kb.create_task(conn, title="live root", assignee="planner")
        kb.add_notify_sub(
            conn,
            task_id=root,
            platform="telegram",
            chat_id="chat-1",
            thread_id="topic-1",
            user_id="user-1",
            notifier_profile="default",
        )
    finally:
        conn.close()

    runner = _make_kanban_slash_runner()
    output = asyncio.run(
        runner._handle_kanban_command(
            _kanban_event(f"/kanban create 'ordinary child' --assignee worker --parent {root}")
        )
    )
    child = _created_id(output)

    assert "subscribed" not in output.lower()
    conn = kb.connect()
    try:
        assert kb.list_notify_subs(conn, child) == []
    finally:
        conn.close()


def test_kanban_create_root_reports_repaired_subscription(tmp_path, monkeypatch):
    db_path = tmp_path / "slash-root-subscribe.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    runner = _make_kanban_slash_runner()
    output = asyncio.run(
        runner._handle_kanban_command(
            _kanban_event("/kanban create 'entry root' --assignee planner")
        )
    )
    root = _created_id(output)

    assert "subscribed" in output.lower()
    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, root)
    finally:
        conn.close()
    assert len(subs) == 1
    assert subs[0]["platform"] == "telegram"
    assert subs[0]["chat_id"] == "chat-1"
    assert subs[0]["thread_id"] == "topic-1"
    assert subs[0]["notifier_profile"] == "default"


def test_kanban_create_completed_parent_child_reports_inherited_subscription(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "slash-completed-parent-child-subscribe.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        root = kb.create_task(conn, title="done root", assignee="planner")
        kb.add_notify_sub(
            conn,
            task_id=root,
            platform="telegram",
            chat_id="chat-1",
            thread_id="topic-1",
            user_id="user-1",
            notifier_profile="default",
        )
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (root,))
        assert kb.complete_task(conn, root, summary="root done")
    finally:
        conn.close()

    runner = _make_kanban_slash_runner()
    output = asyncio.run(
        runner._handle_kanban_command(
            _kanban_event(f"/kanban create 'follow up' --assignee worker --parent {root}")
        )
    )
    child = _created_id(output)

    assert "subscribed" in output.lower()
    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, child)
    finally:
        conn.close()
    assert len(subs) == 1
    assert subs[0]["platform"] == "telegram"
    assert subs[0]["chat_id"] == "chat-1"
    assert subs[0]["thread_id"] == "topic-1"
    assert subs[0]["notifier_profile"] == "default"


def test_repair_root_notify_sub_only_subscribes_roots_and_is_idempotent(tmp_path, monkeypatch):
    db_path = tmp_path / "root-sub-repair.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        root = kb.create_task(conn, title="entry", assignee="planner")
        assert kb.repair_root_notify_sub(
            conn,
            task_id=root,
            platform="telegram",
            chat_id="chat-1",
            thread_id="topic-1",
            user_id="user-1",
            notifier_profile="gateway-a",
        ) is True
        assert kb.repair_root_notify_sub(
            conn,
            task_id=root,
            platform="telegram",
            chat_id="chat-1",
            thread_id="topic-1",
            user_id="user-1",
            notifier_profile="gateway-a",
        ) is True
        child = kb.create_task(conn, title="child", assignee="worker", parents=[root])
        assert kb.repair_root_notify_sub(
            conn,
            task_id=child,
            platform="telegram",
            chat_id="chat-1",
            thread_id="topic-1",
            user_id="user-1",
            notifier_profile="gateway-a",
        ) is False
        root_subs = kb.list_notify_subs(conn, root)
        child_subs = kb.list_notify_subs(conn, child)
    finally:
        conn.close()

    assert len(root_subs) == 1
    assert root_subs[0]["notifier_profile"] == "gateway-a"
    assert child_subs == []


def test_kanban_notifier_escalates_root_only_child_block_to_root_subscriber(tmp_path, monkeypatch):
    """A root-only subscriber must see a child review/block decision.

    Child tasks often have no direct notify row. Without an ancestor/root
    escalation path, root subscribers miss child ``blocked`` /
    ``review-required`` decisions and the workflow waits silently.
    """
    db_path = tmp_path / "root-child-escalation.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    root, child = _create_root_only_blocked_child(
        "review-required: verify token=SECRET123 before merge"
    )

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert root in text
    assert child in text
    assert "review-required" in text
    assert "SECRET123" not in text
    assert "token=" not in text


def test_kanban_notifier_child_escalation_is_idempotent_across_ticks(tmp_path, monkeypatch):
    db_path = tmp_path / "root-child-escalation-dedup.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    _root, child = _create_root_only_blocked_child()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    runner._running = True
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert child in adapter.sent[0]["text"]


def test_kanban_notifier_child_escalation_retries_after_send_failure(tmp_path, monkeypatch):
    db_path = tmp_path / "root-child-escalation-retry.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    _root, child = _create_root_only_blocked_child()

    failing = FailingAdapter()
    runner = _make_runner(failing)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert failing.attempts == 1

    recording = RecordingAdapter()
    runner.adapters = {Platform.TELEGRAM: recording}
    runner._running = True
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(recording.sent) == 1
    assert child in recording.sent[0]["text"]
