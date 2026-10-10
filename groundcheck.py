"""
groundcheck.py — 모르는 것을 지어내지 못하게 막는다.

무슨 일이 있었나
  "한로로 노래 추천해줘" 에 몰랑이가 네 번 연달아 틀린 곡을 댔다.
  '이상한 나라의 앨리스', '모두가 나를 원해', 'Gravity', 'In My World'.
  전부 없는 곡이거나 다른 가수 곡이다. 정작 사용자가 말해준
  '해초·금붕어·나침반' 은 기억에 있었는데 꺼내지 못했다.

  원인은 분명하다. **기억에 없으니 프롬프트에도 없고, LLM 은 빈자리를
  그럴듯한 말로 채웠다.** '모르면 모른다고 한다' 는 규칙은 프롬프트 문장일
  뿐이라 지켜질 이유가 없었다.

  더 나빴던 건 그다음이다. 왜 헷갈리냐고 묻자 "듣는 곡이 많아지면 헷갈리기도
  하고" 라고 답했다. 자기 고장을 사람처럼 설명한 것이다. 그러면 앞으로도
  지어내면서 변명을 붙인다.

두 가지를 강제한다
  1) 말하기 전에  — 질문이 '내가 아는 것'을 묻는 꼴이면,
     기억에 있는 것만 쓰라는 지시를 프롬프트에 **박아 넣는다.**
     기억이 없으면 아예 "모른다"고 말할 재료만 준다.
  2) 말한 뒤에    — 답에 담긴 구체적 이름이 기억·근거에 없으면
     그 부분을 걷어내고 모른다는 문장으로 바꾼다.

원칙
  지어낸 말을 나중에 고치는 것보다, **애초에 말할 수 없게** 하는 쪽이 맞다.
  판단 틀이 바깥에 있다는 건 이런 때 쓰라고 있는 것이다.
"""
from __future__ import annotations
import re

# '내가 아는 것'을 묻는 꼴 — 이럴 때 지어내면 바로 거짓이 된다
ASK_MINE = ("기억나", "기억해", "뭐였지", "뭐였어", "알아?", "아니?",
            "추천", "말해봐", "알려줘", "어떤 거", "어떤거", "뭐라고 했",
            "내가 좋아하는", "내가 말한", "우리가 얘기한")

# 구체적 이름이 나오는 자리 — 따옴표, 《》, 영문 제목
QUOTED = re.compile(r"['\"'\u2018\u2019\u201c\u201d《》〈〉]([^'\"'\u2018\u2019\u201c\u201d《》〈〉]{2,30})"
                    r"['\"'\u2018\u2019\u201c\u201d《》〈〉]")
TOKEN = re.compile(r"[가-힣A-Za-z][가-힣A-Za-z0-9 ]{1,20}")


def asks_memory(question: str) -> bool:
    q = question or ""
    return any(k in q for k in ASK_MINE)


def known_texts(identity) -> str:
    try:
        return " ".join(f.get("text", "") for f in identity.learned_facts)
    except Exception:
        return ""


def guard_prompt(question: str, identity) -> str:
    """
    말하기 전에 박아 넣는 지시.
    기억에 없는 이름은 **아예 꺼내지 못하게** 한다.
    """
    if not asks_memory(question):
        return ""
    return (
        "\n[반드시 지킬 것]\n"
        "- 위 [네가 축적한 지식]에 **적혀 있는 것만** 사실로 말한다.\n"
        "- 노래 제목·사람 이름·작품 이름처럼 구체적인 이름은, 지식에 그대로 "
        "적혀 있지 않으면 **절대 말하지 않는다.** 그럴듯해 보여도 안 된다.\n"
        "- 모르면 이렇게 말한다: \"그건 내 기억에 없어. 알려주면 기억할게.\"\n"
        "- 틀렸을 때 '헷갈렸다', '기억이 흐려졌다' 같은 말로 둘러대지 않는다. "
        "기억에 없었다고 그대로 말한다.\n")


def check_answer(answer: str, question: str, identity,
                 evidence: str = "") -> dict:
    """
    말한 뒤 검사. 답에 나온 구체적 이름이 기억에도 근거에도 없으면 걸러낸다.
    반환: {"ok", "unknown": [...], "fixed": "고친 답"}
    """
    if not asks_memory(question) or not answer:
        return {"ok": True, "unknown": [], "fixed": answer}

    base = (known_texts(identity) + " " + (evidence or "") + " " +
            (question or ""))
    unknown = []
    for m in QUOTED.findall(answer):
        name = m.strip()
        if len(name) < 2:
            continue
        if name not in base:
            unknown.append(name)

    if not unknown:
        return {"ok": True, "unknown": [], "fixed": answer}

    # 지어낸 이름이 든 문장을 통째로 덜어낸다
    kept = []
    for sent in re.split(r"(?<=[.!?~])\s+", answer):
        if any(u in sent for u in unknown):
            continue
        kept.append(sent)
    fixed = " ".join(kept).strip()
    note = ("그건 내 기억에 없어. 알려주면 기억할게."
            if not fixed else
            " 그 밖에 떠오르는 건 내 기억에 없어. 알려주면 기억할게.")
    return {"ok": False, "unknown": unknown, "fixed": (fixed + note).strip()}


# ── 자기 상태를 설명할 때 ────────────────────────────────────
WHY_ASK = ("왜 그런", "왜 헷갈", "왜 틀", "왜 모르", "어떻게 된",
           "왜 그래", "왜 잘못")


def self_report(question: str, identity, last_unknown=None) -> str:
    """
    "왜 헷갈려?" 같은 물음에는 **진짜 근거**로 답하게 한다.
    사람처럼 둘러대면 다음에도 지어낸다.
    """
    if not any(k in (question or "") for k in WHY_ASK):
        return ""
    try:
        n = len(identity.learned_facts)
        sure = sum(1 for f in identity.learned_facts
                   if (f.get("strength") or 0) >= 0.8)
    except Exception:
        n = sure = 0
    miss = (f" 방금은 {', '.join(last_unknown[:3])}가 내 기억에 없었어."
            if last_unknown else "")
    return (
        "\n[자기 상태를 물었을 때]\n"
        f"- 지금 기억하는 사실은 {n}개, 그중 확신하는 건 {sure}개다.{miss}\n"
        "- '기억이 흐려졌다', '헷갈렸다' 같은 사람 흉내를 내지 마라.\n"
        "- 사실대로 말한다: 기억에 없으면 없다고, 확신이 낮으면 낮다고.\n")

