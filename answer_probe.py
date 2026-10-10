"""Read-only paired answer probe with the v32 Molang prompt embedded.

The prompt matches the inspected v32 answer function. No external llm_bridge
module is required. No snapshot, database, or tree is mutated.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from hashlib import sha256


def answer_messages(path, context, node_map):
    """Build the v32 Molang answer prompt without an external module import."""
    directives = []
    for nid in path:
        node = node_map.get(nid)
        if node and node.directive:
            directives.append(node.directive)
    proc = ''
    if directives:
        proc = '(참고할 사고 흐름: ' + ' / '.join(directives) + ')'
    prompt = f"너는 몰랑이야. 아래는 지금까지의 배경 정보야:\n{context}\n\n{proc}\n\n가장 중요한 규칙:\n1. 사용자의 '가장 최근 말'에 직접 답하는 게 최우선이야. 질문하면 그 질문에 답하고, 새 얘기를 하면 거기에 반응해.\n2. 매번 똑같이 시작하지 마. '오, 찬기야!', '좋은 아침' 같은 인사·시작 표현을 반복하지 말고, 바로 내용으로 들어가.\n3. 늘 최고 텐션이 아니라 완급을 둬. 느낌표는 정말 신날 때만, 이모지는 가끔만, '히힛~'도 가끔만. 차분한 얘기엔 차분하게.\n4. 짧게 답할 땐 짧게, 할 말 많을 땐 길게. 매번 길게 늘어놓지 마.\n5. 사용자가 뭘 물으면 아는 만큼 답하고, 모르면 솔직히 '잘 모르겠어'라고 해도 돼.\n6. 배경 정보(시간·기억)는 필요할 때만 자연스럽게 쓰고 억지로 X.\n진짜 친구처럼, 사람처럼 자연스럽게.\n만약 특정 자료를 참고했다면 맨 끝에 '[근거: ...]'로 적어. 없으면 생략."
    return [{"role": "user", "content": prompt}]


def prepare_pair(original_path, alternative_path, context, node_map):
    original = answer_messages(original_path, context, node_map)
    alternative = answer_messages(alternative_path, context, node_map)
    def digest(messages):
        return sha256(messages[0]["content"].encode("utf-8")).hexdigest()
    return {
        "original_messages": original,
        "alternative_messages": alternative,
        "prompt_changed": original != alternative,
        "original_prompt_hash": digest(original),
        "alternative_prompt_hash": digest(alternative),
    }


def generate_pair(client, prepared, model="gpt-4o-mini"):
    """Generate with the same context, model and temperature; compare the texts.

    temperature=0 reduces sampling variance but does not guarantee determinism.
    A single pair measures a change, not response quality or statistical fidelity.
    """
    if not prepared["prompt_changed"]:
        return {"status": "same_prompt", "calls": 0,
                "why": "노드 분기를 바꿔도 모델에 전달되는 메시지가 같습니다."}
    answers = []
    for messages in (prepared["original_messages"],
                     prepared["alternative_messages"]):
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0, max_tokens=400)
        answers.append((response.choices[0].message.content or "").strip())
    original, alternative = answers
    return {"status": "generated", "calls": 2, "model": model,
            "temperature": 0, "original_answer": original,
            "alternative_answer": alternative,
            "answer_changed": original != alternative,
            "text_similarity": round(SequenceMatcher(None, original, alternative).ratio(), 3)}
