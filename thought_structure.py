"""
thought_structure.py — 사고 구조 아키텍처 핵심 엔진.

NM 판단 분기 트리 + LLM 이동 + 대화 반응 학습.

구조:
  - 판단 노드: 하나의 의미 있는 판단 지점
  - 전이 확률(NM): 이 판단에서 다음 판단으로 갈 확률
  - LLM: 각 분기점에서 후보 중 선택 (NM=지도, LLM=운전자)
  - 평가: 대화 반응(긍정/부정)으로 전이 확률 강화/약화

핵심 철학:
  - 처음엔 오류 있어도 됨
  - 대화하며 실수에서 점점 벗어남
  - 학습된 전이 확률 = 정체성
  - 경로 기록 = XAI

정직한 범위:
  - LLM 선택은 주입식(choose_fn). 없으면 확률 기반 자동 선택(시뮬).
  - 이건 뼈대다. 실제 LLM 연결 + 대화 반응 감지는 위에 얹는다.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict
import random
import json
import pickle


@dataclass
class JudgmentNode:
    """판단 노드 — 하나의 의미 있는 판단 지점."""
    id: str
    prompt: str          # 이 판단의 이름 (분기 선택 시 LLM에게 보임)
    directive: str = ""  # 답변 생성 시 실제로 강제되는 구체적 실행 지시
    is_terminal: bool = False   # 최종 판단(잎)인가


@dataclass
class PathRecord:
    """한 번의 사고 궤적 기록 (감사 로그)."""
    path: list                  # 지나간 노드 id 순서
    choices: list               # 각 분기점에서의 선택
    answer: str = ""
    feedback: Optional[float] = None   # 대화 반응 (+1 긍정 / -1 부정 / None)
    context: str = ""           # 입력 질문 (감사: 무엇에 답했나)
    sources: list = field(default_factory=list)  # 가져온 근거 자료 (감사: 무엇을 근거로)
    timestamp: str = ""         # 시각 (감사: 언제)
    verified: bool = True       # 자기검증 통과 여부
    # 재현을 위한 기록 — 경로가 남아도 '그때 무엇을 보고 그 길을 골랐나' 가
    # 없으면 다시 돌릴 수 없다. 기록이 남은 것과 재현되는 것은 다르다.
    at_probs: list = field(default_factory=list)   # 각 갈림길의 그때 확률
    seed: Optional[int] = None                     # 무작위 선택에 쓴 씨앗


class ThoughtStructure:
    """
    NM 판단 분기 트리. LLM이 그 위를 이동하며 사고한다.
    전이 확률이 대화 반응으로 강화/약화 = 학습 = 정체성 성장.
    """
    def __init__(self, learning_rate: float = 0.1,
                 continuity: float = 0.8):
        self.nodes: Dict[str, JudgmentNode] = {}
        # 전이 확률(NM): {from_id: {to_id: prob}}
        self.transitions: Dict[str, Dict[str, float]] = {}
        self.root_id: str = ""
        self.lr = learning_rate            # 학습률 (전이 변화 폭)
        self.continuity = continuity       # 정체성 유지 강도 (뼈대 보존)
        self.history: List[PathRecord] = []
        # 의미 기억 (기억흐름 + 망각 루프)
        # 각 기억: {content, trust, strength, context, timestamp}
        #   trust: 1.0 TRUST_HUMAN(교정/승인) / 0.6 DERIVED / 0.2 DOUBTED
        #   strength: 망각 대상 (안 쓰이면 감소, 쓰이면 회복)
        self.memory: List[dict] = []
        # 대화 맥락 (episodic) — 예전 selfloop 방식. 흐름 유지용.
        #   각 항목: {q, a, timestamp}
        self.episodic: List[dict] = []
        # 양심 (정체성 통합 관문) — 지연 생성
        self._conscience = None

    def _get_conscience(self):
        if self._conscience is None:
            from conscience import Conscience
            self._conscience = Conscience()
        return self._conscience

    # ── 트리 구성 ──
    def add_node(self, node: JudgmentNode, is_root=False):
        self.nodes[node.id] = node
        self.transitions.setdefault(node.id, {})
        if is_root:
            self.root_id = node.id

    def add_branch(self, from_id: str, to_id: str, prob: float = None):
        """판단 분기 추가. prob 없으면 균등 분배."""
        self.transitions.setdefault(from_id, {})
        self.transitions[from_id][to_id] = prob if prob is not None else 0.5
        self._normalize(from_id)

    def _normalize(self, node_id: str):
        """전이 확률 합을 1로."""
        tr = self.transitions.get(node_id, {})
        s = sum(tr.values())
        if s > 0:
            for k in tr:
                tr[k] /= s

    # ── LLM 이동 (사고) ──
    def traverse(self, choose_fn: Optional[Callable] = None,
                 answer_fn: Optional[Callable] = None,
                 context: str = "", seed: Optional[int] = None) -> PathRecord:
        """
        트리를 따라 이동하며 사고.
        choose_fn(node, candidates, probs, context) -> chosen_id
          없으면 전이 확률로 자동 선택(시뮬).
        answer_fn(path, context) -> answer  (최종 답변, 선택)

        seed 를 주면 무작위 선택이 결정적이 되어 **같은 입력으로 같은 경로**가
        나온다. 기록에 그때의 갈림길 확률과 씨앗이 함께 남으므로,
        나중에 그 결정을 다시 돌려보거나 "그때 이 값이 달랐다면" 을
        확인할 수 있다 (replay / counterfactual).
        """
        import random as _rnd
        rng = _rnd.Random(seed) if seed is not None else _rnd
        path, choices, at_probs = [], [], []
        cur = self.root_id
        depth = 0
        while cur and depth < 20:
            path.append(cur)
            node = self.nodes[cur]
            if node.is_terminal:
                break
            candidates = list(self.transitions.get(cur, {}).keys())
            if not candidates:
                break
            probs = [self.transitions[cur][c] for c in candidates]

            # LLM이 선택 (없으면 확률 기반 자동)
            if choose_fn is not None:
                chosen = choose_fn(node, candidates,
                                   [self.nodes[c] for c in candidates],
                                   probs, context)
            else:
                chosen = rng.choices(candidates, weights=probs)[0]

            # 그때의 갈림길 확률을 함께 남긴다 (재현용)
            at_probs.append({"at": cur,
                             "options": {c: round(float(p), 4)
                                         for c, p in zip(candidates, probs)},
                             "chose": chosen})
            choices.append({"at": cur, "chose": chosen})
            cur = chosen
            depth += 1

        answer = ""
        sources = []
        if answer_fn is not None:
            result = answer_fn(path, context)
            # answer_fn이 (답변, 근거리스트) 튜플이면 분리, 아니면 답변만
            if isinstance(result, tuple) and len(result) == 2:
                answer, sources = result
            else:
                answer = result

        from datetime import datetime
        rec = PathRecord(path=path, choices=choices, answer=answer,
                         context=context, sources=sources or [],
                         timestamp=datetime.now().isoformat(timespec="seconds"),
                         at_probs=at_probs, seed=seed)
        return rec

    # ── 되돌려 다시 돌리기 (replay / counterfactual) ──
    def replay(self, record: PathRecord, what_if: dict = None) -> dict:
        """
        기록만으로 그 결정을 **다시 돌린다.**
        what_if 를 주면 그때의 갈림길 확률을 바꿔 "그랬다면" 을 확인한다.
          what_if = {"노드id": {"선택지id": 새확률, ...}}

        반환: {"same": 같은 경로였나, "path": 다시 돈 경로,
               "diverged_at": 갈라진 지점, "original": 원래 경로}
        """
        import random as _rnd
        if not record.at_probs:
            return {"same": None, "why": "그때의 갈림길 확률이 기록에 없음"}

        rng = _rnd.Random(record.seed) if record.seed is not None else _rnd.Random(0)
        path, diverged = [], None
        for i, step in enumerate(record.at_probs):
            at = step["at"]
            opts = dict(step["options"])
            if what_if and at in what_if:
                opts.update(what_if[at])          # 그때 이 값이 달랐다면
            keys = list(opts)
            weights = [max(0.0, float(opts[k])) for k in keys]
            if sum(weights) <= 0:
                break
            chose = rng.choices(keys, weights=weights)[0]
            path.append(at)
            if chose != step["chose"] and diverged is None:
                diverged = {"at": at, "was": step["chose"], "now": chose}
            if diverged:
                # 갈라졌으면 그 뒤는 원래 기록과 다른 길이다
                path.append(chose)
                break
            path.append(chose)

        same = diverged is None
        return {"same": same, "path": path, "diverged_at": diverged,
                "original": list(record.path)}

    # ── 자기검증 (옛 설계 self_verify) ──
    def verify(self, record: PathRecord) -> dict:
        """
        생성된 답이 기존 확정 기억(정체성)과 모순되지 않는지 검증.
        옛 설계의 self_verify(G_t)를 오늘 구조로: 답 내기 전 자기 일관성 점검.
        반환: {"ok": bool, "reason": str, "conflicts": list}
        """
        answer = record.answer or ""
        if not answer or not self.memory:
            return {"ok": True, "reason": "검증 대상 없음", "conflicts": []}

        import re
        ans_words = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', answer))
        conflicts = []
        for m in self.memory:
            if m.get("trust", 0) < 1.0:   # 확정 기억만 기준
                continue
            mem_words = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', m["content"]))
            overlap = ans_words & mem_words
            if not overlap:
                continue
            # 답이 확정 기억과 같은 주제를 다루는데, 값이 모순되나
            ans_only = ans_words - mem_words
            mem_only = mem_words - ans_words
            if overlap and mem_only:
                # 기억의 핵심값이 답에 없고, 다른 값이 있으면 모순 가능성
                value_variant = any(
                    len(a) >= 2 and len(mv) >= 2 and a[:2] == mv[:2]
                    for a in ans_only for mv in mem_only)
                topic_strength = len(overlap) / (min(len(ans_words), len(mem_words)) + 1e-9)
                if topic_strength > 0.3 and mem_only and not value_variant:
                    # 확정 기억의 값이 답에서 누락/모순
                    if not (mem_only & ans_words):
                        conflicts.append(m["content"][:40])
        ok = len(conflicts) == 0
        return {
            "ok": ok,
            "reason": "일관됨" if ok else f"확정 기억과 모순 {len(conflicts)}건",
            "conflicts": conflicts,
        }

    def traverse_verified(self, choose_fn=None, answer_fn=None,
                          context="", max_retry: int = 1) -> PathRecord:
        """
        자기검증 포함 사고 (옛 설계: 검증 실패 시 recalculate).
        답 생성 → 검증 → 모순이면 재사고(재계산). max_retry회까지.
        """
        rec = self.traverse(choose_fn=choose_fn, answer_fn=answer_fn, context=context)
        for attempt in range(max_retry):
            v = self.verify(rec)
            if v["ok"]:
                rec.verified = True
                return rec
            # 검증 실패 → 재계산 (기억을 더 강하게 상기시켜 다시)
            retry_context = (
                f"{context}\n\n[주의: 다음 확정 사실과 모순되지 않게 답하라: "
                f"{', '.join(v['conflicts'])}]")
            rec = self.traverse(choose_fn=choose_fn, answer_fn=answer_fn,
                                context=retry_context)
        rec.verified = self.verify(rec)["ok"]
        return rec

    # ── 대화 반응으로 학습 (전이 강화/약화) ──
    def learn(self, record: PathRecord, feedback: float):
        """
        feedback: +1(긍정) ~ -1(부정)
        지나간 경로의 전이를 강화(긍정) 또는 약화(부정).
        continuity로 뼈대는 보존 (정체성 유지).
        """
        record.feedback = feedback
        self.history.append(record)

        # 경로상의 각 전이를 업데이트
        for ch in record.choices:
            frm, to = ch["at"], ch["chose"]
            tr = self.transitions[frm]
            old = tr.get(to, 0.5)
            # 긍정이면 이 선택 강화, 부정이면 약화
            delta = self.lr * feedback
            # continuity: 급변 방지 (뼈대 보존)
            new = old + delta * (1 - self.continuity) + delta * self.continuity * 0.3
            tr[to] = max(0.01, min(0.99, new))
            self._normalize(frm)

    # ── 의미 기억 (기억흐름 + 망각) ──
    def remember(self, content: str, trust: float = 0.6, context: str = "",
                 use_conscience: bool = True):
        """
        기억할 가치 있는 것만 저장. 아무거나 다 저장 안 함 (망각 철학).
        trust: 1.0 인간교정/승인 / 0.6 파생 / 0.2 의심
        같은 내용 있으면 신뢰·강도만 갱신.

        use_conscience=True면, 확정 기억(trust>=1.0)은 양심 관문을 거친다.
        기존 정체성과 충돌하면 격리(소화 보류) — 분열 방지.
        반환: 실제 저장됐으면 True, 격리/거부면 False.
        """
        from datetime import datetime
        content = content.strip()
        if not content:
            return False
        # 중복 체크 (간단히 포함 관계)
        for m in self.memory:
            if content in m["content"] or m["content"] in content:
                m["trust"] = max(m["trust"], trust)
                m["strength"] = 1.0   # 다시 언급됐으니 회복
                return True

        # 양심 관문 (정체성 축이 되는 확정 기억만)
        if use_conscience and trust >= 1.0:
            verdict = self._get_conscience().evaluate(content, self.memory)
            if verdict.action == "quarantine":
                return False   # 소화 대기 — 아직 정체성에 안 넣음
            if verdict.action == "reject":
                return False   # 정체성 급변 위험 — 거부

        self.memory.append({
            "content": content, "trust": trust, "strength": 1.0,
            "context": context,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        return True

    def remember_from_correction(self, record: PathRecord, prev_record: PathRecord):
        """
        교정 사건(부정→긍정)에서 확정 기억 추출.
        사용자가 고쳐준 것 = TRUST_HUMAN(1.0). 제일 강한 기억.
        """
        # 교정된 답변의 근거·내용을 인간 승인 기억으로
        if record.sources:
            for s in record.sources:
                self.remember(s, trust=1.0, context=record.context)
        # 답변 자체도 짧게 기억 (교정 후 확정된 것)
        if record.answer:
            snippet = record.answer[:100]
            self.remember(snippet, trust=1.0, context=record.context)

    def recall(self, query: str, embed_fn=None, top_k: int = 3) -> list:
        """
        현재 질문에 관련된 기억을 불러옴 (다음 답변에 주입용).
        embed_fn 있으면 의미 유사도, 없으면 단어 겹침.
        신뢰·강도 높은 것 우선.
        """
        if not self.memory:
            return []
        scored = []
        for m in self.memory:
            if embed_fn is not None:
                import numpy as np
                qv, mv = embed_fn(query), embed_fn(m["content"])
                sim = float(np.dot(qv, mv) /
                            ((np.linalg.norm(qv)*np.linalg.norm(mv))+1e-12))
            else:
                import re
                # 조사 차이 극복: 앞 2글자 어간 매칭도 인정
                qw = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', query))
                mw = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', m["content"]))
                hits = 0
                for q in qw:
                    for w in mw:
                        if q == w or (len(q) >= 2 and len(w) >= 2 and q[:2] == w[:2]):
                            hits += 1
                            break
                sim = hits / (len(qw)+1e-9) if qw else 0
            # 관련도 × 신뢰 × 강도
            score = sim * m["trust"] * m["strength"]
            if score > 0.05:
                scored.append((score, m))
        scored.sort(key=lambda x: -x[0])
        # 불러온 기억은 강도 회복 (썼으니까)
        picked = [m for _, m in scored[:top_k]]
        for m in picked:
            m["strength"] = min(1.0, m["strength"] + 0.1)
        return picked

    def forget_step(self, decay: float = 0.05, floor: float = 0.1):
        """
        망각 루프: 안 쓰인 기억은 강도 감소. 바닥 밑이면 제거.
        단, TRUST_HUMAN(1.0)은 잘 안 잊음 (인간 승인은 오래 간다).
        """
        survivors = []
        for m in self.memory:
            # 인간 승인 기억은 망각 저항
            d = decay * (0.3 if m["trust"] >= 1.0 else 1.0)
            m["strength"] -= d
            if m["strength"] > floor:
                survivors.append(m)
        self.memory = survivors

    def memory_context(self, query: str, embed_fn=None) -> str:
        """다음 답변 생성에 넣을 기억 문자열."""
        recalled = self.recall(query, embed_fn=embed_fn)
        if not recalled:
            return ""
        lines = []
        for m in recalled:
            tag = "확정" if m["trust"] >= 1.0 else "참고"
            lines.append(f"[{tag}] {m['content']}")
        return "이전에 확인된 것:\n" + "\n".join(lines)

    # ── 대화 맥락 (episodic) — 흐름 유지 ──
    def add_episode(self, question: str, answer: str):
        """대화 한 턴을 맥락에 쌓는다 (예전 selfloop 방식)."""
        from datetime import datetime
        self.episodic.append({
            "q": question[:120], "a": answer[:200],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        # 너무 길어지면 오래된 것 정리 (최근 30턴만 원문 유지)
        if len(self.episodic) > 30:
            self.episodic = self.episodic[-30:]

    def recent_context(self, n: int = 6) -> str:
        """최근 n턴의 대화 흐름을 프롬프트용 문자열로."""
        if not self.episodic:
            return ""
        recent = self.episodic[-n:]
        lines = [f"사용자: {e['q']}\n나: {e['a']}" for e in recent]
        return "최근 대화 흐름:\n" + "\n---\n".join(lines)

    # ── 정체성 지표 ──
    def dominant_path(self) -> list:
        """현재 가장 강한 판단 경로 = 이 정체성의 기본 사고."""
        path = [self.root_id]
        cur = self.root_id
        seen = {cur}
        while True:
            tr = self.transitions.get(cur, {})
            if not tr:
                break
            nxt = max(tr, key=tr.get)
            if nxt in seen:
                break
            path.append(nxt)
            seen.add(nxt)
            cur = nxt
            if self.nodes[cur].is_terminal:
                break
        return path

    def continuity_rate(self) -> float:
        """최근 궤적들이 얼마나 일관된가 = 정체성 안정도."""
        recent = self.history[-10:]
        if len(recent) < 2:
            return 1.0
        paths = [tuple(r.path) for r in recent]
        most = max(set(paths), key=paths.count)
        return paths.count(most) / len(paths)

    # ── 저장/복원 (정체성 지속) ──
    def save(self, path: str):
        blob = {
            "nodes": {k: v.__dict__ for k, v in self.nodes.items()},
            "transitions": self.transitions,
            "root_id": self.root_id,
            "lr": self.lr, "continuity": self.continuity,
            "history": [r.__dict__ for r in self.history],
            "memory": self.memory,
            "episodic": self.episodic,
            "quarantine": self._conscience.quarantine if self._conscience else [],
        }
        with open(path, "wb") as f:
            pickle.dump(blob, f)
        return path

    @classmethod
    def load(cls, path: str) -> "ThoughtStructure":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        ts = cls(learning_rate=blob["lr"], continuity=blob["continuity"])
        for k, v in blob["nodes"].items():
            ts.nodes[k] = JudgmentNode(**v)
        ts.transitions = blob["transitions"]
        ts.root_id = blob["root_id"]
        ts.history = [PathRecord(**r) for r in blob["history"]]
        ts.memory = blob.get("memory", [])   # 구버전 호환
        ts.episodic = blob.get("episodic", [])  # 대화 맥락 복원
        q = blob.get("quarantine", [])   # 양심 격리소 복원
        if q:
            ts._get_conscience().quarantine = q
        return ts
