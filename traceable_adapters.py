"""
traceable.adapters — 흔한 시스템의 기록을 Trace 로 옮긴다.

어댑터는 얇게 둔다. 시스템마다 자료 모양이 다르니, 여기서는
'어느 필드가 무엇에 해당하는지'만 정해주면 된다.
"""
from __future__ import annotations

from traceable_core import Trace


def from_generic(units_raw, items_raw, state=None, *,
                 unit_used="used", unit_path="path_logged",
                 unit_evidence="evidence", unit_branches="branches",
                 item_accepted="accepted", item_source="source",
                 item_text="text", label="", step=0, size=0,
                 commitments=None, action_events=None,
                 events_complete=False) -> Trace:
    """필드 이름만 알려주면 옮겨준다."""
    units = [{
        "id": u.get("id"),
        "used": bool(u.get(unit_used)),
        "path_logged": bool(u.get(unit_path)),
        "evidence": int(u.get(unit_evidence) or 0),
        "branches": u.get(unit_branches) or [],
    } for u in (units_raw or [])]
    items = [{
        "id": i.get("id"),
        "accepted": bool(i.get(item_accepted)),
        "source": i.get(item_source),
        "text": i.get(item_text) or "",
    } for i in (items_raw or [])]
    return Trace(units=units, items=items, state=dict(state or {}),
                 commitments=list(commitments or []),
                 action_events=list(action_events or []),
                 events_complete=events_complete is True,
                 label=label, step=step, size=size)


def from_langgraph(run_log: list, *, label="", step=0) -> Trace:
    """
    LangGraph 류: 노드를 지난 기록.
    경로는 남지만 근거·학습이 안 남는 경우가 많다 — 그게 그대로 점수에 나온다.
    """
    seen = {}
    for ev in (run_log or []):
        nid = ev.get("node") or ev.get("name")
        if not nid:
            continue
        u = seen.setdefault(nid, {"id": nid, "used": True,
                                  "path_logged": True, "evidence": 0,
                                  "branches": []})
        if ev.get("sources") or ev.get("documents"):
            u["evidence"] += len(ev.get("sources") or ev.get("documents"))
    return Trace(units=list(seen.values()), items=[], state={},
                 label=label, step=step)


def from_rag(chunks: list, *, answer_used=None, label="", step=0) -> Trace:
    """
    RAG: 참조 문서는 남지만 '어떤 경로로 판단했나'가 없다.
    path 가 None 으로 나오는 게 정상이다 — 잴 단위가 없다는 뜻.
    """
    items = [{"id": c.get("id"), "accepted": True,
              "source": c.get("source") or c.get("url"),
              "text": c.get("text") or ""} for c in (chunks or [])]
    return Trace(units=[], items=items, state={}, label=label, step=step,
                 size=len(items))


def from_molang(snapshot: dict, *, label="", step=None) -> Trace:
    """
    몰랑이 성장 기록(pkl 을 읽어 만든 dict) → Trace.
    이 어댑터가 '이 도구가 어디서 나왔는지' 보여주는 예이기도 하다.
    """
    trees = {k: v for k, v in (snapshot.get("trees") or {}).items()
             if isinstance(v, dict)}
    units = []
    for tid, t in trees.items():
        used = (t.get("history_n") or 0) > 0 or (t.get("memory_n") or 0) > 0
        branches = []
        for frm, tos in (t.get("transitions") or {}).items():
            if isinstance(tos, dict) and len(tos) >= 2:
                branches.append({"from": frm, "options": tos})
        units.append({"id": tid, "used": used,
                      "path_logged": (t.get("history_n") or 0) > 0,
                      "evidence": t.get("memory_n") or 0,
                      "branches": branches})

    items = []
    for o in (snapshot.get("observations") or []):
        if not isinstance(o, dict):
            continue
        items.append({"id": o.get("id"), "accepted": o.get("status") == "candidate",
                      "source": o.get("url") or o.get("source_trust"),
                      "text": f"{o.get('topic','')} {o.get('title') or ''}"})
    for f in (snapshot.get("facts") or []):
        items.append({"id": f.get("id"), "accepted": True,
                      "source": f.get("source") or "conversation",
                      "text": f.get("text") or ""})

    # 확신하는 것들 — 연속성을 재는 데 쓴다
    beliefs = {}
    for f in (snapshot.get("facts") or []):
        txt = (f.get("text") or "").strip()
        if txt:
            beliefs[txt] = float(f.get("strength") or 0)

    meta = snapshot.get("meta") or {}
    return Trace(units=units, items=items,
                 state=dict(snapshot.get("interests") or {}),
                 beliefs=beliefs,
                 commitments=list(snapshot.get("commitments") or []),
                 action_events=list(snapshot.get("action_events") or []),
                 events_complete=snapshot.get("events_complete") is True,
                 label=label, step=int(step or meta.get("cycle") or 0),
                 size=len(snapshot.get("facts") or []))
