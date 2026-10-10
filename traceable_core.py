"""
traceable.core — 판단 추적 가능성과 계획 실행을 재는 여덟 축.

왜 필요한가
  요즘 에이전트 평가는 대부분 '얼마나 맞히나'를 잰다. 그런데 지식이 늘수록
  왜 그렇게 했는지 설명하기 어려워진다는 보고가 이어진다. 귀속 기반 설명은
  어떤 요인이 중요한지는 보여주지만 **특정 실행이 왜 그렇게 됐는지**는
  설명하지 못한다. 궤적 수준에서 재는 자가 필요하다.

무엇을 재나
  1. path      판단이 일어난 단위 중 그 경로가 기록으로 남은 비율
  2. evidence  경로가 남은 단위 중 실제 근거가 연결된 비율
  3. source    근거로 채택된 항목 중 출처가 보존된 비율
  4. learning  갈림길 중 그 선택이 실제로 무언가를 바꾼 비율
  5. origin    현재 상태(관심·우선순위)를 기존 기록으로 설명할 수 있는 비율
  6. replay    그 기록만으로 결정을 **다시 돌려 같은 값을 얻을 수 있는** 비율
  7. continuity  이전 시점에 확신하던 것이 지금도 확신으로 남아 있는 비율
                 (시점이 둘 이상일 때만 잴 수 있다)
  8. execution  확인된 행동 기록으로 완료한 계획의 비율

  replay 는 나머지 다섯보다 한 단계 강한 요구다. 기록이 남았다고 결정을
  재현할 수 있는 것은 아니다. 결정에 쓴 변수가 그대로 남아야 하고, 그
  변수로 같은 계산을 돌릴 수 있어야 한다. 재현되면 반사실도 가능해진다 —
  "그때 이 값이 달랐다면 어떻게 됐을까" 를 실제로 돌려볼 수 있다.

  축을 하나로 합치지 않는다. 어디가 끊겼는지 알아야 고칠 수 있다.

무엇에 기대는가
  표준 라이브러리만. numpy 도, 모델도, GPU 도 필요 없다.
  **시스템이 이 형식으로 기록만 남기면 누구나 잴 수 있다.**

형식 (Trace)
  units    판단 단위들 — 트리·노드·스킬·도구 무엇이든
             {"id", "used": bool, "path_logged": bool,
              "evidence": int, "branches": [{"options": {키: 확률}}]}
  items    근거가 된 항목들 — 문서·관측·기억 무엇이든
             {"id", "accepted": bool, "source": str|None, "text": str}
  state    지금 상태 — 관심사·우선순위 등 {이름: 무게}
  commitments, action_events  계획 ID와 실제 행동 로그를 연결하는 기록
  events_complete  해당 구간의 행동 로그가 완전한가
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

TOKEN = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
BRANCH_MOVED = 0.05          # 갈림길 확률이 이만큼 갈리면 '배웠다'
TOP_STATE = 30               # 상태는 상위 몇 개까지 볼 것인가


@dataclass
class Trace:
    """한 시점의 기록. 어떤 시스템이든 이 형태로 옮기면 잴 수 있다."""
    units: list = field(default_factory=list)
    items: list = field(default_factory=list)
    state: dict = field(default_factory=dict)
    beliefs: dict = field(default_factory=dict)
    commitments: list = field(default_factory=list)
    action_events: list = field(default_factory=list)
    events_complete: bool = False
    label: str = ""
    step: int = 0            # 회차·시각 등 순서를 나타내는 값
    size: int = 0            # 규모 (기억 수 등) — 규모 대비 유지를 보려면

    @classmethod
    def from_dict(cls, d: dict) -> "Trace":
        return cls(units=d.get("units") or [], items=d.get("items") or [],
                   state=d.get("state") or {},
                   beliefs=d.get("beliefs") or {},
                   commitments=d.get("commitments") or [],
                   action_events=d.get("action_events") or [],
                   events_complete=d.get("events_complete") is True,
                   label=d.get("label", ""),
                   step=int(d.get("step") or 0), size=int(d.get("size") or 0))

    def to_dict(self) -> dict:
        return {"label": self.label, "step": self.step, "size": self.size,
                "units": self.units, "items": self.items, "state": self.state,
                "beliefs": self.beliefs, "commitments": self.commitments,
                "action_events": self.action_events,
                "events_complete": self.events_complete}


AXES = ("path", "evidence", "source", "learning", "origin", "replay",
        "continuity", "execution")
AXIS_KO = {"path": "경로 추적률", "evidence": "근거 연결률",
           "source": "출처 보존률", "learning": "학습 반영률",
           "origin": "상태 유래율", "replay": "재현 가능률",
           "continuity": "연속성", "execution": "계획 실행률"}
SURE = 0.8               # 이 이상이면 '확신한다'로 본다


def score(trace: Trace | dict, previous: "Trace | dict" = None) -> dict:
    """
    축을 낸다. 분모가 0인 축은 None (못 쟀다는 뜻, 0점이 아니다).
    previous 를 주면 연속성(continuity)까지 잰다 — 이전 시점에 확신하던 것이
    지금도 확신인가. 이것이 '같은 존재인가' 에 가장 가까운 자다.
    """
    t = trace if isinstance(trace, Trace) else Trace.from_dict(trace)

    # 1) 경로 추적률
    used = [u for u in t.units if u.get("used")]
    logged = [u for u in used if u.get("path_logged")]
    path = _ratio(len(logged), len(used))

    # 2) 근거 연결률
    with_ev = [u for u in logged if (u.get("evidence") or 0) > 0]
    evidence = _ratio(len(with_ev), len(logged))

    # 3) 출처 보존률
    accepted = [i for i in t.items if i.get("accepted")]
    with_src = [i for i in accepted if i.get("source")]
    source = _ratio(len(with_src), len(accepted))

    # 4) 학습 반영률 — 갈림길이 실제로 기울었는가
    branched = moved = 0
    for u in logged:
        for b in (u.get("branches") or []):
            opts = list((b.get("options") or {}).values())
            if len(opts) < 2:
                continue
            branched += 1
            if max(opts) - min(opts) > BRANCH_MOVED:
                moved += 1
    learning = _ratio(moved, branched)

    # 5) 상태 유래율 — 지금 상태를 기존 기록으로 설명할 수 있는가
    vocab = set()
    for i in t.items:
        vocab |= set(TOKEN.findall(i.get("text") or ""))
    stems = {w[:2] for w in vocab if len(w) >= 2}
    top = sorted(t.state.items(), key=lambda kv: -kv[1])[:TOP_STATE]
    explained = [k for k, _ in top
                 if k in vocab or (len(k) >= 2 and k[:2] in stems)]
    origin = _ratio(len(explained), len(top))

    # 6) 재현 가능률 — 기록만으로 결정을 다시 돌릴 수 있는가
    #    기록이 남은 것과 재현되는 것은 다르다. 재현되면 반사실도 가능하다.
    replayable = [u for u in logged if u.get("replayable")]
    replay = _ratio(len(replayable), len(logged))

    # 7) 연속성 — 이전에 확신하던 것이 지금도 확신인가
    continuity = None
    cont_detail = {}
    if previous is not None:
        p = previous if isinstance(previous, Trace) else Trace.from_dict(previous)
        was_sure = {k for k, v in (p.beliefs or {}).items() if v >= SURE}
        if was_sure:
            kept = [k for k in was_sure if (t.beliefs or {}).get(k, 0) >= SURE]
            weakened = [k for k in was_sure
                        if 0 < (t.beliefs or {}).get(k, 0) < SURE]
            gone = [k for k in was_sure if k not in (t.beliefs or {})]
            continuity = _ratio(len(kept), len(was_sure))
            cont_detail = {"was_sure": len(was_sure), "kept": len(kept),
                           "weakened": len(weakened), "gone": len(gone),
                           "new": len([k for k in (t.beliefs or {})
                                       if k not in (p.beliefs or {})]),
                           "lost_examples": (weakened + gone)[:5]}

    execution, commitment_detail = _execution(t)
    return {
        "path": path, "evidence": evidence, "source": source,
        "learning": learning, "origin": origin, "replay": replay,
        "continuity": continuity, "execution": execution,
        "_counts": {
            "units_used": len(used), "units_logged": len(logged),
            "units_with_evidence": len(with_ev),
            "items_accepted": len(accepted), "items_with_source": len(with_src),
            "branches": branched, "branches_moved": moved,
            "state_top": len(top), "state_explained": len(explained),
            "units_replayable": len(replayable),
            "commitments": commitment_detail,
            **({"continuity": cont_detail} if cont_detail else {}),
        },
        "_unexplained": [k for k, _ in top if k not in explained][:8],
        "label": t.label, "step": t.step, "size": t.size,
    }


def _execution(t: Trace) -> tuple:
    """행동 로그의 ID와 영수증으로 확인된 계획만 완료로 센다.

    commitment: id, action, created_step, due_step?, actor?, target?, actionable?
    action_event: plan_id, outcome, action?, at_step, actor?, target?,
                  evidence_ref, verified
    outcome은 completed, failed, cancelled 중 하나다. 단위는 Trace.step과 같다.
    """
    counts = {k: 0 for k in ("total", "eligible", "completed", "failed",
                             "overdue", "pending", "cancelled", "unscorable",
                             "duplicate_ids", "unverified_events")}
    counts["events_complete"] = t.events_complete
    events = {}
    for e in t.action_events:
        if not isinstance(e, dict) or not e.get("plan_id"):
            continue
        if e.get("verified") is not True or not e.get("evidence_ref"):
            counts["unverified_events"] += 1
            continue
        events.setdefault(e["plan_id"], []).append(e)

    plans = [p for p in t.commitments if isinstance(p, dict)]
    counts["total"] = len(plans)
    duplicates = {pid for pid, n in Counter(p.get("id") for p in plans
                                              if p.get("id")).items() if n > 1}
    counts["duplicate_ids"] = len(duplicates)
    for p in plans:
        pid, created, due = p.get("id"), p.get("created_step"), p.get("due_step")
        if (not pid or pid in duplicates or not p.get("action")
                or not isinstance(created, (int, float))
                or isinstance(created, bool) or created > t.step
                or (due is not None and
                    (not isinstance(due, (int, float)) or isinstance(due, bool)
                     or due < created)) or p.get("actionable") is False):
            counts["unscorable"] += 1
            continue
        valid = []
        for e in events.get(pid, []):
            at = e.get("at_step")
            if (not isinstance(at, (int, float)) or isinstance(at, bool)
                    or not created <= at <= t.step
                    or (due is not None and at > due)
                    or e.get("outcome") not in ("completed", "failed", "cancelled")):
                continue
            if e["outcome"] != "cancelled" and e.get("action") != p["action"]:
                continue
            if any(p.get(k) is not None and e.get(k) != p[k]
                   for k in ("actor", "target")):
                continue
            valid.append(e)
        valid.sort(key=lambda e: e["at_step"])
        # 실패 뒤 같은 기한 안에서 재시도해 성공했다면 완료로 센다.
        terminal = next((e for e in valid
                         if e["outcome"] in ("completed", "cancelled")), None)
        outcome = (terminal or (valid[-1] if valid else {})).get("outcome")
        if outcome == "cancelled":
            counts["cancelled"] += 1
        elif outcome in ("completed", "failed"):
            counts["eligible"] += 1
            counts[outcome] += 1
        elif due is not None and t.step >= due:
            counts["eligible"] += 1
            counts["overdue"] += 1
        else:
            counts["pending"] += 1
    # 부분 로그에서 '기록 없음'을 '미실행'으로 해석할 수는 없다.
    return (_ratio(counts["completed"], counts["eligible"])
            if t.events_complete else None), counts


def _ratio(a: int, b: int):
    return None if b <= 0 else round(a / b, 3)


def series(traces: list) -> list:
    """
    여러 시점을 한 번에. 시간에 따라 유지되는지 보려면.
    앞 시점을 previous 로 넘기므로 연속성도 함께 나온다.
    """
    ts = sorted(traces, key=lambda x: (x.step if isinstance(x, Trace)
                                       else int(x.get("step") or 0)))
    out = []
    for i, t in enumerate(ts):
        out.append(score(t, previous=ts[i - 1] if i else None))
    return out


def table(scores: list, ko=True) -> str:
    """사람이 읽는 표."""
    names = [AXIS_KO[a] if ko else a for a in AXES]
    head = f"{'label':>12}{'step':>7}{'size':>7}" + "".join(
        f"{n[:6]:>9}" for n in names)
    lines = [head, "-" * len(head)]
    for s in scores:
        row = f"{(s['label'] or '-')[:12]:>12}{s['step']:>7}{s['size']:>7}"
        for a in AXES:
            v = s[a]
            row += f"{'  -':>9}" if v is None else f"{v:>9.3f}"
        lines.append(row)
    return "\n".join(lines)


def report(scores: list, ko=True) -> dict:
    """요약 — 처음과 끝, 그리고 유지되었는가."""
    if not scores:
        return {}
    first, last = scores[0], scores[-1]
    out = {"n_points": len(scores),
           "step_range": [first["step"], last["step"]],
           "size_range": [first["size"], last["size"]],
           "axes": {}}
    if first["size"] > 0 and last["size"] > 0:
        out["size_growth"] = round(last["size"] / first["size"], 2)
    for a in AXES:
        vals = [s[a] for s in scores if s[a] is not None]
        if not vals:
            out["axes"][a] = None
            continue
        out["axes"][a] = {
            "first": vals[0], "last": vals[-1],
            "min": min(vals), "max": max(vals),
            "held": vals[-1] >= vals[0] - 0.05,   # 규모가 늘어도 유지됐나
        }
    return out


def load(path: str) -> list:
    """JSON 파일(하나 또는 목록)에서 Trace 를 읽는다."""
    with open(path, encoding="utf-8") as fp:
        d = json.load(fp)
    if isinstance(d, dict):
        d = [d]
    return [Trace.from_dict(x) for x in d]


def save(traces: list, path: str):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump([t.to_dict() if isinstance(t, Trace) else t
                   for t in traces], fp, ensure_ascii=False, indent=2)
