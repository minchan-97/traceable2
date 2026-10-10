"""Read-only paired answer probe using the bundled Molang answer function's prompt.

This copies the prompt through the actual make_answer_fn rather than maintaining
another prompt template. No snapshot, database, or tree is mutated.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from hashlib import sha256
from types import SimpleNamespace

from llm_bridge import make_answer_fn


class _CaptureCompletions:
    def __init__(self):
        self.messages = None

    def create(self, **kwargs):
        self.messages = kwargs["messages"]
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="[capture]"))])


def answer_messages(path, context, node_map):
    """Return precisely the messages assembled by Molang's bundled answer_fn."""
    completions = _CaptureCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    make_answer_fn(client, node_map)(path, context)
    if completions.messages is None:
        raise RuntimeError("답변 프롬프트를 포착하지 못했습니다.")
    return completions.messages


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
