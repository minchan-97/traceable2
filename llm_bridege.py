"""
llm_bridge.py — 사고 구조에 실제 LLM 연결.

세 가지 연결:
  1. choose_fn: 분기점에서 LLM이 어느 판단으로 갈지 선택
  2. answer_fn: 지나온 경로를 바탕으로 최종 답변 생성
  3. detect_feedback: 사람의 대화 반응(긍정/부정) 감지

로컬(Ollama)/GPT 둘 다 지원 — base_url만 다름.
  GPT:    OpenAI(api_key=...)
  Ollama: OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
"""
from __future__ import annotations
from typing import Callable, List, Optional
import re


def make_client(api_key: str = "", local: bool = False):
    from openai import OpenAI
    if local:
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    key = "".join(ch for ch in api_key.strip() if ord(ch) < 128)
    return OpenAI(api_key=key)


def make_choose_fn(client, model: str = "gpt-4o-mini"):
    """
    분기점에서 LLM이 판단을 선택.
    NM 전이 확률을 참고로 주되, 최종 선택은 LLM이 (NM=지도, LLM=운전자).
    """
    def choose_fn(node, candidates, candidate_nodes, probs, context):
        # 후보 판단들을 LLM에게 제시.
        #
        # 자생으로 생긴 갈래(grown_*)는 설명이 빈약하다. 그러면 LLM 이
        # **늘 첫 번째 기본 갈래만** 고르고, 전이 확률이 학습한 것이
        # 쓰이질 않는다 (실제로 0.37 짜리를 두고 0.22 짜리로 네 번 갔다).
        # 그래서 설명이 없으면 이름이라도 보여주고, 성향이 높은 것을
        # 앞에 놓아 눈에 먼저 들어오게 한다.
        order = sorted(range(len(candidates)), key=lambda i: -probs[i])
        candidates = [candidates[i] for i in order]
        candidate_nodes = [candidate_nodes[i] for i in order]
        probs = [probs[i] for i in order]

        def _label(cn, cid):
            t = (getattr(cn, "prompt", "") or "").strip()
            if len(t) < 4:
                t = (getattr(cn, "name", "") or cid).replace("_", " ")
            return t[:70]

        options = "\n".join(
            f"  {i+1}. {_label(cn, cid)} (지금까지 이 길로 {p:.0%})"
            for i, (cn, cid, p) in enumerate(
                zip(candidate_nodes, candidates, probs)))
        prompt = (
            f"맥락: {context}\n\n"
            f"현재 판단 지점: {node.prompt}\n\n"
            f"다음 중 어느 판단으로 진행할지 하나만 고르세요:\n{options}\n\n"
            f"번호만 답하세요 (1~{len(candidates)}). "
            f"지금까지의 비율은 **이 상황에서 대개 어느 길로 갔는지**를 "
            f"뜻한다. 맥락이 다르면 다른 길을 골라도 되지만, "
            f"특별한 이유가 없으면 그 비율을 따르세요.")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=10)
            text = resp.choices[0].message.content.strip()
            m = re.search(r'\d+', text)
            if m:
                idx = int(m.group()) - 1
                if 0 <= idx < len(candidates):
                    return candidates[idx]
        except Exception:
            pass
        # 실패 시 확률 최대 후보
        return candidates[probs.index(max(probs))]
    return choose_fn


def make_answer_fn(client, node_map, model: str = "gpt-4o-mini"):
    """
    지나온 판단 경로의 directive(구체적 실행 지시)를 순서대로 실행.
    절차는 지키되, 답변은 자연스러운 대화체로.
    반환: (답변, 근거자료 리스트) — 근거는 감사 기록용.
    """
    def answer_fn(path, context):
        directives = []
        for nid in path:
            node = node_map.get(nid)
            if node and node.directive:
                directives.append(node.directive)

        proc = ""
        if directives:
            proc = "(참고할 사고 흐름: " + " / ".join(directives) + ")"

        prompt = (
            f"너는 몰랑이야. 아래는 지금까지의 배경 정보야:\n"
            f"{context}\n\n"
            f"{proc}\n\n"
            f"가장 중요한 규칙:\n"
            f"1. 사용자의 '가장 최근 말'에 직접 답하는 게 최우선이야. "
            f"질문하면 그 질문에 답하고, 새 얘기를 하면 거기에 반응해.\n"
            f"2. 매번 똑같이 시작하지 마. '오, 찬기야!', '좋은 아침' 같은 "
            f"인사·시작 표현을 반복하지 말고, 바로 내용으로 들어가.\n"
            f"3. 늘 최고 텐션이 아니라 완급을 둬. 느낌표는 정말 신날 때만, "
            f"이모지는 가끔만, '히힛~'도 가끔만. 차분한 얘기엔 차분하게.\n"
            f"4. 짧게 답할 땐 짧게, 할 말 많을 땐 길게. 매번 길게 늘어놓지 마.\n"
            f"5. 사용자가 뭘 물으면 아는 만큼 답하고, 모르면 솔직히 "
            f"'잘 모르겠어'라고 해도 돼.\n"
            f"6. 배경 정보(시간·기억)는 필요할 때만 자연스럽게 쓰고 억지로 X.\n"
            f"진짜 친구처럼, 사람처럼 자연스럽게.\n"
            f"만약 특정 자료를 참고했다면 맨 끝에 '[근거: ...]'로 적어. 없으면 생략."
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7, max_tokens=400)
            text = resp.choices[0].message.content.strip()
            sources = []
            import re
            for m in re.finditer(r'\[근거:\s*([^\]]+)\]', text):
                sources.append(m.group(1).strip())
            # 근거 표기는 답변에서 떼어내 따로 (답변은 깔끔하게)
            clean = re.sub(r'\[근거:\s*[^\]]+\]', '', text).strip()
            return clean, sources
        except Exception as e:
            return f"(답변 생성 실패: {e})", []
    return answer_fn


# ── 대화 반응 감지 ──
_NEGATIVE = ['아니', '틀렸', '아냐', '아닌', '잘못', '다시', '이상해', '별로',
             '아니오', '노', 'no', 'wrong', '그게 아니', '안 맞', '틀림']
_POSITIVE = ['맞아', '맞다', '좋아', '좋다', '정확', '그래', '응', '옳', '완벽',
             '고마', '훌륭', 'yes', 'good', '맞습니다', '좋습니다']


def detect_feedback(user_message: str,
                    client=None, model: str = "gpt-4o-mini") -> Optional[float]:
    """
    사람의 반응에서 긍정/부정 감지.
    1차: 명백한 표현 (규칙)
    2차: 애매하면 LLM (client 있을 때)
    반환: +1(긍정) / -1(부정) / None(중립·불명)
    """
    msg = user_message.strip().lower()

    # 1차: 명백한 표현
    neg = any(w in msg for w in _NEGATIVE)
    pos = any(w in msg for w in _POSITIVE)
    if neg and not pos:
        return -1.0
    if pos and not neg:
        return 1.0

    # 2차: 애매하면 LLM (선택)
    if client is not None:
        try:
            prompt = (
                f"사용자 반응: \"{user_message}\"\n\n"
                f"이 반응이 직전 답변에 만족(긍정)인지 불만족(부정)인지 중립인지 판단하세요.\n"
                f"positive / negative / neutral 중 하나만 답하세요.")
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=10)
            t = resp.choices[0].message.content.strip().lower()
            if "positive" in t:
                return 1.0
            if "negative" in t:
                return -1.0
        except Exception:
            pass
    return None   # 중립·불명 → 학습 안 함


# ── 정보 제공 감지 (사용자가 "기억해둬" 하는 것) ──
def detect_shared_info(user_message: str,
                       client=None, model: str = "gpt-4o-mini") -> str:
    """
    사용자가 자기 정보를 알려주는지 감지 → 기억할 내용 반환.
    "내 이름은 민찬기야" → "사용자 이름은 민찬기"
    질문이거나 정보가 아니면 "" 반환.
    """
    msg = user_message.strip()
    # 질문이면 정보 제공 아님 (질문을 기억으로 저장하는 오류 방지)
    question_markers = ['뭐', '무엇', '누구', '어디', '언제', '왜', '어떻', '?',
                        '뭘', '몇', '했지', '했더라', '이야?', '야?', '까?']
    if any(qm in msg for qm in question_markers):
        return ""

    # 1차: 명백한 패턴 (규칙)
    import re
    patterns = [
        (r'^내?\s*이름(은|는)\s*([가-힣a-zA-Z]{2,})(이야|야|입니다|이에요|예요)?$',
         lambda m: f"사용자 이름은 {m.group(2)}"),
        (r'나는?\s*(.+?)(를|을)\s*좋아', lambda m: f"사용자는 {m.group(1)}을(를) 좋아함"),
        (r'나는?\s*(.+?)(이|가)\s*싫', lambda m: f"사용자는 {m.group(1)}을(를) 싫어함"),
        (r'내?\s*(취미|직업|나이|고향|사는\s*곳)(은|는)\s*(.+?)(이야|야|입니다|이에요|예요|있어)?$',
         lambda m: f"사용자 {m.group(1)}: {m.group(3)}"),
    ]
    for pat, fn in patterns:
        m = re.search(pat, msg)
        if m:
            return fn(m).strip()

    # 2차: LLM 판별 (애매할 때)
    if client is not None:
        try:
            prompt = (
                f"사용자 메시지: \"{user_message}\"\n\n"
                f"이 메시지에서 '나중에 기억해야 할 사용자 정보'만 뽑아주세요.\n"
                f"규칙:\n"
                f"- 정보가 있으면: 사실만 간결히. 목록이면 항목을 그대로 나열.\n"
                f"- 질문이거나 정보가 없으면: 'NONE'\n"
                f"- 절대 '~이므로', '~를 제공하고 있다', '기억하겠다' 같은 "
                f"설명·판단은 쓰지 마세요. 사실 그 자체만.\n\n"
                f"예: '내 이름은 민찬기야' → '이름: 민찬기'\n"
                f"예: '내 시계는 롤렉스, 오메가, 세이코야' → '시계: 롤렉스, 오메가, 세이코'\n"
                f"예: '내 시계 뭐라고 했지?' → 'NONE'\n"
                f"예: '오늘 날씨 어때?' → 'NONE'")
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=150)
            t = resp.choices[0].message.content.strip()
            # 메타 설명이 섞였으면 거른다 (안전장치)
            bad_markers = ["이므로", "제공하고", "기억할", "기억하겠", "따라서",
                           "메시지는", "메시지가", "알려주고 있"]
            if t and "NONE" not in t.upper() and not any(b in t for b in bad_markers):
                return t
        except Exception:
            pass
    return ""


# ── 새 유형 사고 트리 생성 (extract_reasoning_pattern을 LLM으로) ──
def make_tree_designer(client, model: str = "gpt-4o-mini"):
    """
    새 유형이 감지되면, LLM이 그 유형에 맞는 판단 단계(트리)를 설계.
    1년 전 설계의 extract_reasoning_pattern + map_to_function_template를 LLM으로.
    반환: design_fn(question, type_hint) -> {"type_id", "nodes"}
    """
    import json as _json

    def design_fn(question: str, type_hint: str = ""):
        prompt = (
            f"질문: \"{question}\"\n\n"
            f"이 질문에 답하려면 어떤 '판단 단계'를 거쳐야 하는지 사고 절차를 설계하세요.\n"
            f"2~3개의 순차적 판단 단계로 나누고, 각 단계에 구체적 지시를 쓰세요.\n"
            f"JSON만 출력 (설명 없이):\n"
            f'{{"type_id": "유형이름(영문소문자)", '
            f'"steps": [{{"name": "단계명", "directive": "이 단계에서 할 구체적 지시"}}]}}\n\n'
            f"예: 비교 질문 → "
            f'{{"type_id": "compare", "steps": ['
            f'{{"name": "대상 파악", "directive": "무엇과 무엇을 비교하는지 명확히 한다"}}, '
            f'{{"name": "기준 설정", "directive": "어떤 기준으로 비교할지 정한다"}}, '
            f'{{"name": "종합 판단", "directive": "각 기준별 차이를 종합해 결론낸다"}}]}}')
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=300)
            t = resp.choices[0].message.content.strip()
            t = t.replace("```json", "").replace("```", "").strip()
            data = _json.loads(t)
            if "type_id" in data and "steps" in data and data["steps"]:
                return data
        except Exception:
            pass
        return None
    return design_fn


def make_classifier(client, model: str = "gpt-4o-mini"):
    """
    질문을 사고 유형 중 하나로 분류. 정말 새로운 '추론 방식'일 때만 None(→새 유형).
    화제(취향/시간/일상)가 다르다고 새 유형을 만들지 않는다.
    """
    def classify_fn(question, type_ids, type_examples):
        try:
            lines = []
            for tid in type_ids[:40]:   # 더 많이 보여줘 매칭률 높임
                exs = type_examples.get(tid, [])
                ex = exs[0][:28] if exs else ""
                lines.append(f"- {tid}: {ex}")
            catalog = "\n".join(lines)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": (
                    "아래 '말'이 어떤 '추론/사고 방식'인지 목록에서 하나 골라줘.\n\n"
                    "중요 규칙:\n"
                    "- 유형은 '화제(축구/취향/시간)'가 아니라 '사고 방식'이다. "
                    "(예: 비교하기, 원인 추론, 조건 판단, 분류, 예시 들기, 감정 공감 등)\n"
                    "- 일상 대화·잡담·인사·감정 표현은 대부분 기존의 일반적 유형에 속한다. "
                    "웬만하면 목록에서 가장 가까운 것을 고를 것.\n"
                    "- '완전히 새로운 추론 구조'여서 목록 어디에도 안 맞을 때만 'NEW'. "
                    "화제가 새롭다는 이유로 NEW를 쓰지 마라.\n\n"
                    f"[유형 목록]\n{catalog}\n\n"
                    f"[말] {question}\n\n"
                    "유형 id 하나만 (정말 새로운 사고방식이면 NEW):")}],
                temperature=0, max_tokens=15)
            ans = resp.choices[0].message.content.strip()
            if ans in type_ids:
                return ans
            return None
        except Exception:
            return None
    return classify_fn


def make_consolidator(client, model: str = "gpt-4o-mini"):
    """
    응고화 함수. 대화에서 '장기 기억할 압축된 사실'을 뽑고,
    기존 기억과 충돌하는지 판단한다.
    반환: {"fact": "찬기는 커피를 좋아한다", "conflicts_with": "기존사실"|None}
          또는 {"fact": "NONE"} (기억할 것 없음)
    """
    import json
    def consolidate_fn(question, answer, existing_facts):
        try:
            known = "\n".join(f"- {t[:50]}" for t in existing_facts[-15:]) or "(없음)"
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": (
                    "사용자가 **직접 말한** 내용에서만, 사용자에 대해 오래 기억할 "
                    "사실을 뽑아줘.\n"
                    "규칙:\n"
                    "0. 정보의 출처가 '몰랑이의 답변'이면 무조건 fact를 'NONE'으로. "
                    "몰랑이가 자기 얘기를 한 것(무엇을 공부했는지, 무엇을 상상했는지 등)을 "
                    "사용자의 사실로 바꾸지 마라. 사용자가 '너 뭐 했어?'라고 물은 턴은 "
                    "대부분 'NONE'이다.\n"
                    "1. 취향·직업·습관·관계·계획 같은 지속적 사실만.\n"
                    "2. 짧고 명확한 한 문장으로 압축. 예: '찬기는 커피를 좋아한다'\n"
                    "   (수다체 말고 사실만. 인사·잡담이면 fact를 'NONE')\n"
                    "3. 이 사실이 기존 기억 중 하나와 '모순'되면 conflicts_with에 그 기존 사실을 적어.\n"
                    "   (예: 기존 '축구를 좋아한다' vs 새 '축구를 이제 안 본다')\n"
                    "   모순 없으면 conflicts_with는 null.\n\n"
                    f"[기존 기억]\n{known}\n\n"
                    f"[사용자가 한 말 — 여기서만 사실을 뽑는다]\n{question}\n\n"
                    f"[몰랑이가 한 답 — 참고만, 여기서 사실을 뽑지 마라]\n{answer}\n\n"
                    'JSON만 출력: {"fact": "...", "conflicts_with": null 또는 "기존사실"}')}],
                temperature=0, max_tokens=100)
            txt = resp.choices[0].message.content.strip()
            txt = txt.replace("```json", "").replace("```", "").strip()
            return json.loads(txt)
        except Exception:
            return None
    return consolidate_fn
