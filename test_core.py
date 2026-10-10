# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from traceable import Trace, score, series, report
from traceable.adapters import from_molang


def test_empty_axes_are_none():
    """잴 게 없으면 0 이 아니라 None 이어야 한다."""
    s = score(Trace())
    assert all(s[a] is None for a in ("path", "evidence", "source",
                                      "learning", "origin", "execution"))


def test_execution_requires_complete_log_and_real_receipt():
    plans = [
        {"id": "ask-piu", "action": "ask", "actor": "molang",
         "target": "piupiu", "created_step": 1, "due_step": 3},
        {"id": "visit", "action": "visit", "created_step": 1, "due_step": 3},
        {"id": "future", "action": "visit", "created_step": 3, "due_step": 5},
        {"id": "cancel", "action": "search", "created_step": 1, "due_step": 3},
    ]
    events = [
        {"plan_id": "ask-piu", "outcome": "completed", "action": "ask",
         "actor": "molang", "target": "piupiu", "at_step": 2,
         "verified": True, "evidence_ref": "peer_talk:101"},
        {"plan_id": "visit", "outcome": "completed", "action": "visit",
         "at_step": 2, "verified": False, "evidence_ref": "self_report"},
        {"plan_id": "cancel", "outcome": "cancelled", "at_step": 2,
         "verified": True, "evidence_ref": "user_cancel:1"},
    ]
    t = Trace(step=3, commitments=plans, action_events=events)
    assert score(t)["execution"] is None
    t.events_complete = True
    s = score(t)
    assert s["execution"] == 0.5
    assert s["_counts"]["commitments"]["overdue"] == 1
    assert s["_counts"]["commitments"]["pending"] == 1
    assert s["_counts"]["commitments"]["cancelled"] == 1


def test_execution_rejects_wrong_action_target_and_late_event():
    t = Trace(step=5, events_complete=True, commitments=[
        {"id": "p", "action": "ask", "target": "piupiu",
         "created_step": 1, "due_step": 3}], action_events=[
        {"plan_id": "p", "outcome": "completed", "action": "move",
         "target": "piupiu", "at_step": 2, "verified": True,
         "evidence_ref": "move:1"},
        {"plan_id": "p", "outcome": "completed", "action": "ask",
         "target": "someone_else", "at_step": 2, "verified": True,
         "evidence_ref": "talk:2"},
        {"plan_id": "p", "outcome": "completed", "action": "ask",
         "target": "piupiu", "at_step": 4, "verified": True,
         "evidence_ref": "talk:3"}])
    assert score(t)["execution"] == 0.0


def test_molang_legacy_snapshot_cannot_measure_execution():
    t = from_molang({"meta": {"cycle": 490}, "facts": [], "trees": {}})
    assert score(t)["execution"] is None
    assert score(t)["_counts"]["commitments"]["eligible"] == 0


def test_execution_round_trip():
    t = Trace(step=2, commitments=[{"id": "p", "action": "ask",
                                    "created_step": 1, "due_step": 2}],
              action_events=[{"plan_id": "p", "outcome": "failed",
                              "action": "ask", "at_step": 2,
                              "verified": True, "evidence_ref": "action:1"}],
              events_complete=True)
    assert score(Trace.from_dict(t.to_dict()))["execution"] == 0.0


def test_execution_retry_can_succeed_before_deadline():
    t = Trace(step=4, events_complete=True,
              commitments=[{"id": "p", "action": "ask",
                            "created_step": 1, "due_step": 4}],
              action_events=[
                  {"plan_id": "p", "outcome": "failed", "action": "ask",
                   "at_step": 2, "verified": True, "evidence_ref": "attempt:1"},
                  {"plan_id": "p", "outcome": "completed", "action": "ask",
                   "at_step": 3, "verified": True, "evidence_ref": "talk:2"}])
    assert score(t)["execution"] == 1.0


def test_path_ratio():
    t = Trace(units=[{"id": 1, "used": True, "path_logged": True},
                     {"id": 2, "used": True, "path_logged": False},
                     {"id": 3, "used": False, "path_logged": False}])
    assert score(t)["path"] == 0.5          # 쓰인 둘 중 하나만 남음


def test_learning_needs_real_tilt():
    flat = Trace(units=[{"id": 1, "used": True, "path_logged": True,
                         "branches": [{"options": {"a": .5, "b": .5}}]}])
    tilt = Trace(units=[{"id": 1, "used": True, "path_logged": True,
                         "branches": [{"options": {"a": .8, "b": .2}}]}])
    assert score(flat)["learning"] == 0.0
    assert score(tilt)["learning"] == 1.0


def test_source_only_counts_accepted():
    t = Trace(items=[{"id": 1, "accepted": True, "source": "u"},
                     {"id": 2, "accepted": True, "source": None},
                     {"id": 3, "accepted": False, "source": None}])
    assert score(t)["source"] == 0.5


def test_origin_explains_state():
    t = Trace(items=[{"id": 1, "accepted": True, "source": "u",
                      "text": "바다는 넓다"}],
              state={"바다": 2.0, "난데없는말": 1.0})
    s = score(t)
    assert s["origin"] == 0.5
    assert s["_unexplained"] == ["난데없는말"]


def test_replay_is_stricter_than_path():
    """기록이 남은 것과 재현되는 것은 다르다."""
    t = Trace(units=[
        {"id": 1, "used": True, "path_logged": True, "replayable": True},
        {"id": 2, "used": True, "path_logged": True, "replayable": False},
    ])
    s = score(t)
    assert s["path"] == 1.0        # 둘 다 경로는 남았다
    assert s["replay"] == 0.5      # 그런데 하나만 다시 돌릴 수 있다


def test_continuity_needs_two_points():
    """연속성은 시점이 둘이어야 잴 수 있다."""
    a = Trace(beliefs={"바다를 좋아한다": 1.0, "파스타를 안다": 0.9}, step=1)
    b = Trace(beliefs={"바다를 좋아한다": 1.0, "파스타를 안다": 0.5,
                       "새로 안 것": 0.9}, step=2)
    assert score(a)["continuity"] is None          # 혼자서는 못 잰다
    s = score(b, previous=a)
    assert s["continuity"] == 0.5                  # 둘 중 하나만 확신 유지
    assert s["_counts"]["continuity"]["weakened"] == 1


def test_series_and_report():
    mk = lambda step, n: Trace(
        units=[{"id": 1, "used": True, "path_logged": True, "evidence": 1}],
        items=[{"id": i, "accepted": True, "source": "u", "text": "바다"}
               for i in range(n)],
        state={"바다": 1.0}, step=step, size=n)
    r = report(series([mk(10, 5), mk(50, 20)]))
    assert r["size_growth"] == 4.0
    assert r["axes"]["path"]["held"] is True


if __name__ == "__main__":
    import traceback
    ok = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); ok += 1; print(f"  ✓ {name}")
            except Exception:
                print(f"  ✗ {name}"); traceback.print_exc()
    print(f"\n{ok}개 통과")
