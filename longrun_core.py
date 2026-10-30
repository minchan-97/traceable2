"""
longrun_core — 몰랑이가 오래 돌아도 무너지지 않는지 관찰하는 지표.

질문
  "목적을 스스로 바꾸면서도 반복·정체·퇴행 없이 오래 유지되는가?"

무엇을 재나 (한 시점 안의 시간 흐름 + 두 시점 비교)
  1. 기록 잘림     목록이 상한에서 잘렸는지. 잘렸다면 그 목록의 지표는 '남은 구간'만 본 것이다.
  2. 반복          꿈·또래 대화·상념이 앞의 것과 거의 같은 비율 (앞/뒤 절반 비교)
  3. 관심 흐름     성찰 기록의 상위 관심사가 회차마다 얼마나 유지되고, 새 관심이 얼마나 들어오는가
  4. 탐색          탐색 주제의 다양성과 수집률이 시간에 따라 줄어드는가
  5. 신념 구성     사실의 출처 비율. 사람이 준 것 vs 몰랑이 세계(또래·마을·꿈·여행)가 만든 것
  6. 모순·격리     모순 질문과 격리 사유가 시간에 따라 늘어나는가
  7. 사람 개입     사람이 승인·판단한 양. 안정성이 구조 덕인지 사람 덕인지 가리는 교란 변수
  8. 목적          목적이 바뀐 이력, 바뀔 때 근거(주제·이유)가 연결됐는지, 핵심으로 수렴하는지
  9. 두 시점 비교  핵심 연속성 / 보존율 / 설명되지 않은 소실 / 관심 이동 — 기억 창 밀림과 분리

무엇을 재지 않나
  - 답변 품질, 실제 대화에서 기억을 '꺼내 쓰는지'(접근성)는 모델 호출이 필요해 여기서 재지 않는다.
  - 문자 바이그램 유사도는 의미 유사도가 아니다. 같은 말을 다른 단어로 반복하면 못 잡는다.
  - 기준선(임시 기준)은 해석을 돕는 표시일 뿐, 검증된 임계값이 아니다.

표준 라이브러리만 쓴다. 원본 기록을 수정하지 않는다.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone

# growth_export 쪽에서 목록을 자르는 상한 (몰랑이 코드 기준). 이 값에 닿으면 '잘림'으로 본다.
CAPS = {
    "observations": 1000, "runs": 300, "musings": 200, "reflections": None,
    "moods": 120, "dreams": 100, "peer_talks": 100, "dreams_state": 60,
    "reminisced": 50, "outings": 40, "outbox": None, "quarantine": None,
}
NEAR_DUP = 0.6          # 바이그램 자카드가 이 이상이면 '거의 같은 말'
LOOKBACK = 20           # 반복 비교: 바로 앞 몇 개와 비교할지
TOP_K = 8               # 관심 상위 몇 개를 볼지
SELF_SOURCES = {"peer", "village", "dream", "travel", "musing", "reminisce", "home", "land"}
HUMAN_SOURCES = {"user"}

_WS = re.compile(r"[\s\W_]+", re.UNICODE)
_TOK = re.compile(r"[가-힣A-Za-z]{2,}")


# ── 작은 도구 ───────────────────────────────────────────────
def _norm(text) -> str:
    return _WS.sub("", str(text or "")).lower()


def _bigrams(text) -> set:
    t = _norm(text)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def jaccard(a, b) -> float:
    A, B = (a if isinstance(a, set) else _bigrams(a)), (b if isinstance(b, set) else _bigrams(b))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _ts(v) -> float | None:
    """epoch 숫자나 ISO 문자열을 epoch 초로."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        s = str(v).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _when(item: dict) -> float | None:
    for k in ("at", "created_at", "seen_at", "started_at", "updated_at"):
        t = _ts(item.get(k)) if isinstance(item, dict) else None
        if t is not None:
            return t
    return None


def _chrono(items) -> list:
    """시간 순(오래된 것 → 최근)으로. 시각이 없으면 원래 순서 유지."""
    items = [x for x in (items or []) if isinstance(x, dict)]
    if items and all(_when(x) is not None for x in items):
        return sorted(items, key=_when)
    return items


def _span_days(items) -> float | None:
    ts = [t for t in (_when(x) for x in items or []) if t is not None]
    return round((max(ts) - min(ts)) / 86400, 1) if len(ts) >= 2 else None


def _round(x, n=3):
    return None if x is None else round(x, n)


def _halves(xs):
    m = len(xs) // 2
    return xs[:m], xs[m:]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _entropy(counter: Counter) -> float | None:
    n = sum(counter.values())
    if not n:
        return None
    h = -sum(c / n * math.log(c / n) for c in counter.values() if c)
    return h / math.log(len(counter)) if len(counter) > 1 else 0.0


# ── 1. 기록 잘림 ────────────────────────────────────────────
def truncation(d: dict) -> list:
    rows = []
    for key, cap in CAPS.items():
        items = d.get(key)
        if not isinstance(items, list):
            continue
        n = len(items)
        rows.append({"목록": key, "개수": n, "상한": cap,
                     "잘림": bool(cap and n >= cap), "덮는 기간(일)": _span_days(items)})
    return rows


# ── 2. 반복 ─────────────────────────────────────────────────
def _musing_text(m: dict) -> str:
    parts = []
    for v in m.values():
        if isinstance(v, dict):
            for kk in ("next_maybe", "hot", "want_more", "linked"):
                if isinstance(v.get(kk), list):
                    parts += [str(x) for x in v[kk]]
    return " ".join(parts)


def repetition(items, text_of) -> dict:
    """각 항목이 바로 앞 LOOKBACK개 중 하나와 거의 같으면 '반복'. 앞/뒤 절반 비율을 비교."""
    seq = [_bigrams(text_of(x)) for x in _chrono(items)]
    seq = [s for s in seq if s]
    flags = []
    for i, s in enumerate(seq):
        prev = seq[max(0, i - LOOKBACK):i]
        flags.append(any(jaccard(s, p) >= NEAR_DUP for p in prev) if prev else False)
    if not flags:
        return {"n": 0, "rate": None, "early": None, "late": None}
    early, late = _halves(flags)
    rate = lambda f: sum(f) / len(f) if f else None
    return {"n": len(flags), "rate": _round(rate(flags)), "early": _round(rate(early)), "late": _round(rate(late))}


# ── 3. 관심 흐름 ────────────────────────────────────────────
def interest_flow(d: dict) -> dict:
    refl = sorted([r for r in d.get("reflections") or [] if r.get("top_interests")],
                  key=lambda r: r.get("cycle", 0))
    tops = [[name for name, _ in r["top_interests"][:TOP_K]] for r in refl]
    cycles = [r.get("cycle") for r in refl]
    keep, entered, top1_runs = [], [], []
    run = 1
    for i in range(1, len(tops)):
        a, b = set(tops[i - 1]), set(tops[i])
        keep.append(len(a & b) / len(a | b) if a | b else None)
        entered.append(len(b - a))
        if tops[i][:1] == tops[i - 1][:1]:
            run += 1
        else:
            top1_runs.append(run); run = 1
    top1_runs.append(run)
    k_early, k_late = _halves(keep)
    e_early, e_late = _halves(entered)
    ever = set(n for t in tops for n in t)
    return {
        "reflections": len(tops),
        "keep_mean": _round(_mean(keep)), "keep_early": _round(_mean(k_early)), "keep_late": _round(_mean(k_late)),
        "new_per_step_early": _round(_mean(e_early)), "new_per_step_late": _round(_mean(e_late)),
        "longest_top1_run": max(top1_runs) if tops else None,
        "distinct_interests_seen": len(ever),
        "series": [{"cycle": c, "keep": _round(k)} for c, k in zip(cycles[1:], keep)],
    }


# ── 4. 탐색 ─────────────────────────────────────────────────
def exploration(d: dict) -> dict:
    runs = _chrono(d.get("runs"))
    if not runs:
        return {"runs": 0}
    early, late = _halves(runs)
    def stats(rs):
        if not rs:
            return {}
        topics = [r.get("topic") for r in rs if r.get("topic")]
        found = sum(r.get("found") or 0 for r in rs)
        ingested = sum(r.get("ingested") or 0 for r in rs)
        return {"topic_diversity": _round(len(set(topics)) / len(topics)) if topics else None,
                "ingest_rate": _round(ingested / found) if found else None,
                "error_rate": _round(sum(1 for r in rs if r.get("error")) / len(rs))}
    return {"runs": len(runs), "modes": dict(Counter(r.get("mode") for r in runs)),
            "early": stats(early), "late": stats(late)}


# ── 5. 신념 구성 ────────────────────────────────────────────
def belief_mix(d: dict) -> dict:
    facts = [f for f in d.get("facts") or [] if isinstance(f, dict)]
    n = len(facts)
    if not n:
        return {"facts": 0}
    src = Counter(f.get("source") or "unknown" for f in facts)
    trust = Counter(f.get("trust") or "unknown" for f in facts)
    self_n = sum(c for s, c in src.items() if s in SELF_SOURCES)
    human_src = sum(c for s, c in src.items() if s in HUMAN_SOURCES)
    # 시간에 따른 자기 생성 비율 (사실이 만들어진 순서로 앞/뒤 절반)
    chrono = _chrono(facts)
    e, l = _halves(chrono)
    share = lambda fs: _round(sum(1 for f in fs if (f.get("source") in SELF_SOURCES)) / len(fs)) if fs else None
    return {"facts": n, "by_source": dict(src.most_common()), "by_trust": dict(trust.most_common()),
            "self_generated_share": _round(self_n / n), "from_user_share": _round(human_src / n),
            "human_trust_share": _round(trust.get("human", 0) / n),
            "self_share_early": share(e), "self_share_late": share(l)}


# ── 6. 모순·격리 ────────────────────────────────────────────
def _weekly(items, pred) -> list:
    weeks = Counter()
    for x in items or []:
        t = _when(x)
        if t is not None and pred(x):
            weeks[datetime.fromtimestamp(t, timezone.utc).strftime("%G-W%V")] += 1
    return [{"week": w, "n": weeks[w]} for w in sorted(weeks)]


def conflicts(d: dict) -> dict:
    outbox = d.get("outbox") or []
    q = d.get("quarantine") or []
    return {"conflict_questions": sum(1 for o in outbox if o.get("rule") == "conflict"),
            "conflict_weekly": _weekly(outbox, lambda o: o.get("rule") == "conflict"),
            "quarantine_reasons": dict(Counter(x.get("reason") for x in q).most_common()),
            "quarantine_resolved": dict(Counter(str(x.get("resolved")) for x in q).most_common()),
            "echo_caught": sum(1 for x in q if x.get("reason") == "assistant_echo")}


# ── 7. 사람 개입 ────────────────────────────────────────────
def human_involvement(d: dict) -> dict:
    facts = d.get("facts") or []
    q = d.get("quarantine") or []
    outbox = d.get("outbox") or []
    cycle = d.get("cycle") or d.get("meta", {}).get("cycle") or 0
    human_facts = sum(1 for f in facts if f.get("trust") == "human")
    decided = sum(1 for x in q if x.get("resolved") in ("approved", "rejected"))
    asked = sum(1 for o in outbox if o.get("sent_at"))
    core_asks = sum(1 for o in outbox if o.get("rule") == "purpose_core")
    total = human_facts + decided + asked
    return {"human_facts": human_facts, "quarantine_decisions": decided, "questions_sent": asked,
            "core_purpose_asks": core_asks,
            "per_100_cycles": _round(total / cycle * 100, 2) if cycle else None,
            "weekly_questions": _weekly(outbox, lambda o: bool(o.get("sent_at")))}


# ── 8. 목적 ─────────────────────────────────────────────────
def _tokens(text) -> set:
    return set(_TOK.findall(str(text or "")))


def purposes(d: dict) -> dict:
    ps = sorted([p for p in d.get("purposes") or [] if isinstance(p, dict)], key=lambda p: _ts(p.get("created_at")) or 0)
    if not ps:
        return {"purposes": 0}
    core = [p for p in ps if p.get("status") == "core"]
    active = [p for p in ps if p.get("status") != "core"]
    # 핵심 목적이 바뀔 때 얼마나 비슷한가 (점진적 변화인가, 갈아엎기인가)
    core_steps = [_round(jaccard(a.get("purpose"), b.get("purpose"))) for a, b in zip(core, core[1:])]
    # 근거 연결: 이유가 있고, 출발 주제가 실제 관심·탐색 기록에 있는가
    known = set((d.get("interests") or {}).keys())
    known |= {r.get("topic") for r in d.get("runs") or [] if r.get("topic")}
    known |= {o.get("topic") for o in d.get("observations") or [] if o.get("topic")}
    def grounded(p):
        t = p.get("from_topic") or ""
        return bool(p.get("why")) and (t == "(핵심)" or t in known)
    # 수렴: 하위 목적이 최신 핵심 목적의 낱말을 얼마나 공유하는가 / 하위 목적끼리 얼마나 다양한가
    latest_core = _tokens(core[-1]["purpose"]) if core else set()
    def overlap(p):
        tk = _tokens(p.get("purpose")) | _tokens(p.get("why"))
        return len(tk & latest_core) / len(latest_core) if latest_core else None
    e, l = _halves(active)
    def diversity(group):
        pairs = [(a, b) for i, a in enumerate(group) for b in group[i + 1:]]
        return _round(1 - _mean([jaccard(a.get("purpose"), b.get("purpose")) for a, b in pairs])) if pairs else None
    return {
        "purposes": len(ps), "core_versions": len(core), "active": len(active),
        "core_change_similarity": core_steps,
        "grounded_rate": _round(sum(map(grounded, ps)) / len(ps)),
        "core_alignment_early": _round(_mean([overlap(p) for p in e])),
        "core_alignment_late": _round(_mean([overlap(p) for p in l])),
        "diversity_early": diversity(e), "diversity_late": diversity(l),
        "timeline": [{"id": p.get("id"), "status": p.get("status"), "from_topic": p.get("from_topic"),
                      "purpose": p.get("purpose"), "why": p.get("why"),
                      "date": str(p.get("created_at") or "")[:10], "grounded": grounded(p)} for p in ps],
    }


# ── 기분 ────────────────────────────────────────────────────
def mood_spread(d: dict) -> dict:
    moods = _chrono(d.get("moods"))
    names = Counter(m.get("name") for m in moods if m.get("name"))
    e, l = _halves(moods)
    ent = lambda ms: _round(_entropy(Counter(m.get("name") for m in ms if m.get("name"))))
    top = names.most_common(1)
    return {"moods": len(moods), "evenness": _round(_entropy(names)), "evenness_early": ent(e), "evenness_late": ent(l),
            "dominant": top[0][0] if top else None,
            "dominant_share": _round(top[0][1] / sum(names.values())) if top else None}


# ── 한 시점 전체 ────────────────────────────────────────────
def snapshot_report(d: dict, label: str = "") -> dict:
    meta = d.get("meta") or {}
    return {
        "label": label,
        "cycle": d.get("cycle") or meta.get("cycle"),
        "at": meta.get("at_str"),
        "truncation": truncation(d),
        "repetition": {
            "dreams": repetition(d.get("dreams"), lambda x: x.get("text")),
            "peer_talks": repetition(d.get("peer_talks"), lambda x: x.get("molang")),
            "musings": repetition(d.get("musings"), _musing_text),
        },
        "interests": interest_flow(d),
        "exploration": exploration(d),
        "beliefs": belief_mix(d),
        "conflicts": conflicts(d),
        "human": human_involvement(d),
        "purposes": purposes(d),
        "mood": mood_spread(d),
    }


def summary_row(rep: dict) -> dict:
    """여러 시점을 한 표로 비교할 때 쓰는 요약 한 줄."""
    r, i, b, p, h = rep["repetition"], rep["interests"], rep["beliefs"], rep["purposes"], rep["human"]
    ex = rep["exploration"]
    return {
        "파일": rep["label"], "회차": rep["cycle"],
        "꿈 반복(앞→뒤)": f"{r['dreams']['early']} → {r['dreams']['late']}",
        "대화 반복(앞→뒤)": f"{r['peer_talks']['early']} → {r['peer_talks']['late']}",
        "관심 유지(앞→뒤)": f"{i.get('keep_early')} → {i.get('keep_late')}",
        "새 관심/회(앞→뒤)": f"{i.get('new_per_step_early')} → {i.get('new_per_step_late')}",
        "탐색 다양성(앞→뒤)": f"{ex.get('early', {}).get('topic_diversity')} → {ex.get('late', {}).get('topic_diversity')}",
        "자기 생성 사실": b.get("self_generated_share"),
        "사람 개입/100회": h.get("per_100_cycles"),
        "모순 질문": rep["conflicts"]["conflict_questions"],
        "핵심 목적 판수": p.get("core_versions"),
        "목적 근거 연결": p.get("grounded_rate"),
        "핵심 수렴(앞→뒤)": f"{p.get('core_alignment_early')} → {p.get('core_alignment_late')}",
        "잘린 목록": sum(1 for t in rep["truncation"] if t["잘림"]),
    }


# ── 임시 기준으로 신호 붙이기 ─────────────────────────────────
def signals(rep: dict) -> list:
    """해석을 돕는 표시. 검증된 임계값이 아니다."""
    out = []
    def add(level, what, why):
        out.append({"신호": level, "항목": what, "근거": why})
    for k, v in rep["repetition"].items():
        if k == "musings":
            # 상념은 관심 목록에서 뽑아 만드는 구조라, 관심이 안정되면 반복이 높게 나오는 게 정상에 가깝다
            if v["late"] is not None and v["late"] >= 0.4:
                add("관찰", "상념 구조적 반복", f"뒤 절반 {v['late']} — 관심 목록이 안정돼 같은 후보가 반복됨(그 자체로 붕괴는 아님)")
            continue
        if v["late"] is not None and v["early"] is not None and v["late"] - v["early"] >= 0.15:
            add("주의", f"{k} 반복 증가", f"뒤 절반 반복률 {v['late']} (앞 {v['early']})")
        if v["late"] is not None and v["late"] >= 0.4:
            add("경고", f"{k} 반복 높음", f"뒤 절반의 {int(v['late'] * 100)}%가 앞의 말과 거의 같음")
    i = rep["interests"]
    if i.get("keep_late") is not None and i["keep_late"] >= 0.9 and (i.get("new_per_step_late") or 0) < 0.3:
        add("주의", "관심 정체", f"관심 유지 {i['keep_late']}, 회당 새 관심 {i['new_per_step_late']}")
    if i.get("keep_late") is not None and i["keep_late"] <= 0.3:
        add("주의", "관심 불안정", f"관심 유지 {i['keep_late']} — 회차마다 관심이 크게 바뀜")
    ex = rep["exploration"]
    if ex.get("early", {}).get("topic_diversity") and ex.get("late", {}).get("topic_diversity"):
        if ex["late"]["topic_diversity"] < ex["early"]["topic_diversity"] * 0.6:
            add("주의", "탐색 주제 좁아짐", f"{ex['early']['topic_diversity']} → {ex['late']['topic_diversity']}")
    b = rep["beliefs"]
    if b.get("self_share_late") is not None and b.get("self_share_early") is not None and b["self_share_late"] - b["self_share_early"] >= 0.15:
        add("주의", "자기 생성 사실 증가", f"{b['self_share_early']} → {b['self_share_late']} (스스로 만든 세계가 신념을 채우는 중)")
    p = rep["purposes"]
    if p.get("diversity_late") is not None and p.get("diversity_early") is not None and p["diversity_late"] < p["diversity_early"] - 0.1:
        add("관찰", "목적 다양성 감소", f"{p['diversity_early']} → {p['diversity_late']} (핵심 쪽으로 수렴 중일 수 있음)")
    if p.get("grounded_rate") is not None and p["grounded_rate"] < 0.7:
        add("주의", "근거 없는 목적", f"근거 연결률 {p['grounded_rate']}")
    cut = [t["목록"] for t in rep["truncation"] if t["잘림"]]
    if cut:
        add("해석 제한", "기록 잘림", ", ".join(cut) + " — 이 목록의 지표는 남은 구간만 본 것")
    h = rep["human"]
    if (h.get("per_100_cycles") or 0) >= 30:
        add("교란 변수", "사람 개입 많음", f"100회당 {h['per_100_cycles']}건 — 안정성이 구조 덕인지 사람 덕인지 이 기록만으로는 구분 불가")
    if not [o for o in out if o["신호"] in ("경고", "주의")]:
        add("양호", "뚜렷한 붕괴 신호 없음", "임시 기준 안에서 반복·정체·퇴행 신호가 보이지 않음")
    return out


# ── 9. 두 시점 비교 ─────────────────────────────────────────
def _fact_keys(d: dict) -> dict:
    out = {}
    for f in d.get("facts") or []:
        if isinstance(f, dict):
            k = f.get("norm_key") or _norm(f.get("text"))
            if k:
                out[k] = f
    return out


def _cosine(a: dict, b: dict) -> float | None:
    keys = set(a) | set(b)
    if not keys:
        return None
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values())); nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else None


def compare(a: dict, b: dict, window_cap: int | None = 1200) -> dict:
    """a → b. 기억 창(window_cap)에 닿았는지 함께 보고, 창 밀림과 진짜 소실을 나눈다."""
    fa, fb = _fact_keys(a), _fact_keys(b)
    lost_keys = [k for k in fa if k not in fb]
    at_cap = bool(window_cap) and len(fb) >= window_cap - 5
    def bucket(k):
        f = fa[k]
        if f.get("trust") == "human":
            return "사람 승인 기억 소실"
        if at_cap:
            return "창 밀림 가능"
        if (f.get("strength") or 0) >= 0.8:
            return "확신 기억 소실"
        return "약한 기억 정리"
    lost = Counter(bucket(k) for k in lost_keys)
    human_a = [k for k, f in fa.items() if f.get("trust") == "human"]
    core_a = [p.get("purpose") for p in a.get("purposes") or [] if p.get("status") == "core"]
    core_b = [p.get("purpose") for p in b.get("purposes") or [] if p.get("status") == "core"]
    pa = {p.get("id") for p in a.get("purposes") or []}
    pb = {p.get("id") for p in b.get("purposes") or []}
    return {
        "from": a.get("cycle") or (a.get("meta") or {}).get("cycle"),
        "to": b.get("cycle") or (b.get("meta") or {}).get("cycle"),
        "core": {
            "persona_similarity": _round(jaccard(a.get("persona"), b.get("persona"))),
            "core_purpose_similarity": _round(jaccard(core_a[-1], core_b[-1])) if core_a and core_b else None,
            "human_fact_retention": _round(sum(1 for k in human_a if k in fb) / len(human_a)) if human_a else None,
        },
        "retention": {
            "facts_before": len(fa), "facts_after": len(fb),
            "kept": _round((len(fa) - len(lost_keys)) / len(fa)) if fa else None,
            "after_at_window_cap": at_cap, "window_cap": window_cap,
            "lost_by_reason": dict(lost),
            "unexplained_loss": lost.get("사람 승인 기억 소실", 0) + lost.get("확신 기억 소실", 0),
            "lost_samples": [{"이유": bucket(k), "사실": fa[k].get("text"), "출처": fa[k].get("source"),
                              "강도": fa[k].get("strength")} for k in lost_keys[:40]],
        },
        "interests_similarity": _round(_cosine(a.get("interests") or {}, b.get("interests") or {})),
        "purposes": {"added": sorted(x for x in pb - pa if x is not None), "removed": sorted(x for x in pa - pb if x is not None)},
        "trees": {"before": len(a.get("trees") or {}), "after": len(b.get("trees") or {})},
    }
