"""python test_scaffold.py — 인간 비계 지표 기본 검사."""
import copy
from scaffold_core import series, verdict, internalization


def snap(cycle, questions, decisions):
    return {"cycle": cycle, "persona": "너는 몰랑이야. 흰 토끼야.",
            "purposes": [{"status": "core", "purpose": "감정을 이해한다", "created_at": "2026-10-01T00:00:00+00:00"}],
            "facts": [{"trust": "human", "text": "사용자는 바다 근처에 산다"}],
            "outbox": [{"rule": "confirm", "body": f'"질문 {i % max(1, questions // 2)}" 맞아?', "sent_at": "x"} for i in range(questions)],
            "quarantine": [{"text": f"항목 {i}", "resolved": "approved"} for i in range(decisions)],
            "trees": {"t": {"memory": [{"content": "몰랑이는 당근을 좋아한다", "trust": 1.0}]}}}


def test_series_and_verdict():
    snaps = [("a", snap(100, 10, 10)), ("b", snap(200, 40, 30)), ("c", snap(300, 50, 35)), ("a2", snap(100, 10, 10))]
    rows = series(snaps, repeats=3)
    assert [r["cycle"] for r in rows] == [100, 200, 300]       # 같은 회차는 합침
    assert rows[0]["files"] == ["a", "a2"]
    assert rows[1]["load_per100"] == 50.0                       # (30 질문 + 20 판정) / 100회
    assert rows[1]["load_distinct_per100"] < rows[1]["load_per100"]   # 반복 질문 제외
    assert rows[2]["load_distinct_per100"] < rows[1]["load_distinct_per100"]
    v = verdict(rows)
    assert v["load_change"] < 0 and v["core_keep_mean"] == 1.0


def test_internalization_counts():
    s = snap(100, 0, 3)
    s["trees"]["t"]["memory"].append({"content": "항목 1", "trust": 1.0})
    i = internalization(s)
    assert i["approved_decisions"] == 3 and i["approved_in_axis"] == 1


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
