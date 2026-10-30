"""python test_longrun.py — 장기 안정성 지표 기본 검사 (표준 라이브러리만)."""
import copy
from longrun_core import snapshot_report, signals, compare, repetition, jaccard


def base():
    facts = [{"id": i, "text": f"사실 {i}번 내용", "norm_key": f"k{i}", "strength": 0.9 if i < 3 else 0.6,
              "trust": "human" if i == 0 else "derived", "source": "user" if i < 6 else "peer",
              "created_at": f"2026-10-{i + 1:02d}T00:00:00+00:00"} for i in range(10)]
    refl = [{"cycle": c, "top_interests": [[n, 1.0] for n in (["바다", "노래", "하늘"] if c < 5 else ["바다", "노래", "별"])]}
            for c in range(1, 10)]
    purposes = [{"id": 1, "purpose": "사람의 감정을 이해하고 싶다", "from_topic": "(핵심)", "why": "시작", "status": "core",
                 "created_at": "2026-10-01T00:00:00+00:00"},
                {"id": 2, "purpose": "바다가 주는 감정을 알고 싶다", "from_topic": "바다", "why": "바다를 자주 봄", "status": "active",
                 "created_at": "2026-10-02T00:00:00+00:00"},
                {"id": 3, "purpose": "아무 근거 없는 목적", "from_topic": "없는주제", "why": "", "status": "active",
                 "created_at": "2026-10-03T00:00:00+00:00"}]
    return {"cycle": 100, "meta": {"cycle": 100}, "persona": "나는 몰랑이야. 다정하게 말해.",
            "facts": facts, "reflections": refl, "purposes": purposes, "interests": {"바다": 1.0, "노래": 0.5},
            "dreams": [{"text": f"꿈 {i} 서로 다른 이야기 {i * 7}", "created_at": f"2026-10-{i + 1:02d}T00:00:00+00:00"} for i in range(10)],
            "outbox": [{"rule": "conflict", "created_at": "2026-10-05T00:00:00+00:00", "sent_at": "2026-10-05T00:01:00+00:00"}],
            "quarantine": [{"reason": "assistant_echo", "resolved": "rejected"}], "trees": {"t": {}}}


def test_repetition_detects_loop():
    loop = [{"text": "나는 바다에 가서 노래를 불렀다", "at": i} for i in range(10)]
    assert repetition(loop, lambda x: x["text"])["late"] == 1.0
    varied = [{"text": f"완전히 다른 문장 번호 {i} {'가나다라마바사'[i % 7] * 3}", "at": i} for i in range(10)]
    assert repetition(varied, lambda x: x["text"])["rate"] < 0.5


def test_report_runs_and_grounding():
    rep = snapshot_report(base(), "t")
    assert rep["purposes"]["grounded_rate"] == round(2 / 3, 3)   # 근거 없는 목적 1개
    assert rep["beliefs"]["self_generated_share"] == 0.4          # peer 4개 / 10개
    assert rep["interests"]["reflections"] == 9
    assert any(s["항목"] == "근거 없는 목적" for s in signals(rep))


def test_compare_separates_window_and_real_loss():
    a = base(); b = copy.deepcopy(a)
    b["facts"] = [f for f in b["facts"] if f["id"] not in (0, 1, 8)]   # 사람 승인 1, 확신 1, 약한 1 소실
    c = compare(a, b, window_cap=1200)
    assert c["retention"]["lost_by_reason"] == {"사람 승인 기억 소실": 1, "확신 기억 소실": 1, "약한 기억 정리": 1}
    assert c["retention"]["unexplained_loss"] == 2
    c2 = compare(a, b, window_cap=7)    # 이후 시점이 창 상한(7)에 닿음 → 확신 소실도 '창 밀림 가능'
    assert c2["retention"]["lost_by_reason"].get("창 밀림 가능") == 2
    assert c2["retention"]["unexplained_loss"] == 1                   # 사람 승인 기억 소실은 창과 무관하게 문제


def test_jaccard_basic():
    assert jaccard("바다 노래", "바다노래") == 1.0
    assert jaccard("", "무엇") == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
