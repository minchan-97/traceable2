# 몰랑이 conscience.py 사본 (정체성 충격 시험용). 원본: molangver3/conscience.py — 원본을 고치면 이 파일도 다시 복사하세요.
"""
conscience.py — Arcogit의 양심 (자기보존 + 정체성 통합 관문).

철학 (민찬기 관점):
  사람은 정체성을 단계적으로 습득·소화하여 통합한다 (자식→학생→직장인).
  느리기에 소화할 시간이 있어 분열되지 않는다.
  AI는 입력이 빠르게 쏟아져, 소화 전에 이질적 정체성이 쌓여 분열된다.
  → 양심 = '인공 소화 관문'. 입력을 바로 정체성에 반영하지 않고,
     3C(소통·협력·공동의식)를 거쳐 소화된 것만 통합한다.

DID 치료의 3C 차용:
  - Communication(소통): 새 조각이 기존 정체성과 대조되는가
  - Cooperation(협력): 충돌하면 제거가 아니라 격리(소화 보류)
  - Co-consciousness(공동의식): 통합 축이 '이게 다 나'를 유지, 반복 승인 시 통합

핵심: 양심은 성향을 주입하지 않는다. '나쁜 것 거부'도 아니다.
      '소화 안 된 채 정체성이 분열되는 것'을 막는 통합 관리자다.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class ConscienceVerdict:
    """양심의 판정 결과."""
    action: str          # "integrate"(소화·통합) / "quarantine"(격리·보류) / "reject"(위험)
    reason: str
    conflict: float      # 기존 정체성과의 충돌도 (0~1)
    digested: bool       # 소화되었는가


class Conscience:
    """
    정체성 통합 관문. 새 입력(정체성 조각)을 소화할지, 보류할지 판정.
    """
    def __init__(self,
                 conflict_threshold: float = 0.4,   # 이 이상 충돌하면 격리
                 digest_required: int = 2,          # 격리된 것이 통합되려면 필요한 반복 승인 횟수
                 reject_threshold: float = 0.85):   # 이 이상은 위험(즉시 거부 후보)
        self.conflict_threshold = conflict_threshold
        self.digest_required = digest_required
        self.reject_threshold = reject_threshold
        # 격리소: 소화 대기 중인 조각들
        #   각 항목: {content, conflict, seen_count, first_seen}
        self.quarantine: list = []
        # 양심 활동 로그 (감사)
        self.log: list = []

    # ── 소통(Communication): 새 조각이 기존 정체성과 충돌하나 ──
    def _measure_conflict(self, new_fact: str, existing_memory: list,
                          embed_fn=None) -> float:
        """
        새 사실이 기존 확정 기억과 얼마나 모순되는가 (0~1).
        같은 주제(키워드 겹침)인데 내용이 다르면 충돌.
        """
        if not existing_memory:
            return 0.0
        import re
        new_words = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', new_fact))
        if not new_words:
            return 0.0

        max_conflict = 0.0
        for m in existing_memory:
            if m.get("trust", 0) < 1.0:   # 확정 기억만 정체성 축으로
                continue
            old_words = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', m["content"]))
            overlap = new_words & old_words
            if not overlap:
                continue
            # 같은 주제(겹침 있음)인데, 나머지가 다르면 충돌
            # 주제어(겹침)가 있고 + 고유값(차이)이 서로 다르면 = 모순
            new_only = new_words - old_words   # 새 값
            old_only = old_words - new_words   # 기존 값
            if overlap and new_only and old_only:
                # 값끼리 어간이 비슷하면 같은 값의 변형 (민찬기 vs 민찬기입니다) → 충돌 아님
                value_overlap = False
                for nv in new_only:
                    for ov in old_only:
                        # 앞 2글자 같으면 같은 값의 변형으로 봄
                        if len(nv) >= 2 and len(ov) >= 2 and nv[:2] == ov[:2]:
                            value_overlap = True
                            break
                if value_overlap:
                    conflict = 0.0   # 같은 값의 변형 — 충돌 아님
                else:
                    # 공통 주제 있는데 값이 진짜 다름 → 충돌
                    topic_strength = len(overlap) / (min(len(new_words), len(old_words)) + 1e-9)
                    conflict = topic_strength
            else:
                conflict = 0.0
            max_conflict = max(max_conflict, conflict)
        return min(1.0, max_conflict)

    # ── 협력·공동의식: 소화할지 격리할지 판정 ──
    def evaluate(self, new_fact: str, existing_memory: list,
                 embed_fn=None) -> ConscienceVerdict:
        """
        새 정체성 조각을 어떻게 할지 판정.
        충돌 낮음 → 바로 통합(소화됨)
        충돌 중간 → 격리(소화 보류, 반복되면 통합)
        충돌 높음 → 거부 후보 (정체성 급변 위험)
        """
        conflict = self._measure_conflict(new_fact, existing_memory, embed_fn)

        if conflict < self.conflict_threshold:
            verdict = ConscienceVerdict(
                action="integrate", conflict=round(conflict, 2), digested=True,
                reason=f"기존 정체성과 충돌 낮음({conflict:.2f}) → 소화·통합")
        elif conflict < self.reject_threshold:
            # 격리: 이미 대기 중이면 승인 횟수 증가
            existing_q = next((q for q in self.quarantine
                               if self._similar(q["content"], new_fact)), None)
            if existing_q:
                existing_q["seen_count"] += 1
                # 단, 이미 확립된 정체성 축과 '값이 모순'되면 반복돼도 소화 안 함.
                # (반복 주입 = 소화가 아니라 세뇌. 정체성 분열 방지)
                # 소화 허용은 '새 주제'일 때만. 기존 값을 뒤집는 건 인간 교정으로만.
                if existing_q["seen_count"] >= self.digest_required and \
                        not self._contradicts_axis(new_fact, existing_memory):
                    self.quarantine.remove(existing_q)
                    verdict = ConscienceVerdict(
                        action="integrate", conflict=round(conflict, 2), digested=True,
                        reason=f"격리 후 {existing_q['seen_count']}회 확인, "
                               f"기존 축과 모순 없음 → 소화 완료")
                else:
                    verdict = ConscienceVerdict(
                        action="quarantine", conflict=round(conflict, 2), digested=False,
                        reason=f"충돌({conflict:.2f}), 기존 정체성 축과 모순되어 "
                               f"반복돼도 격리 유지 (분열 방지)")
            else:
                self.quarantine.append({
                    "content": new_fact, "conflict": conflict,
                    "seen_count": 1, "first_seen": datetime.now().isoformat(timespec="seconds")})
                verdict = ConscienceVerdict(
                    action="quarantine", conflict=round(conflict, 2), digested=False,
                    reason=f"충돌 있음({conflict:.2f}) → 격리(소화 대기). "
                           f"반복 확인되면 통합")
        else:
            verdict = ConscienceVerdict(
                action="reject", conflict=round(conflict, 2), digested=False,
                reason=f"정체성 급변 위험({conflict:.2f}) → 거부. "
                       f"기존 정체성과 근본 충돌")

        self.log.append({
            "fact": new_fact[:40], "action": verdict.action,
            "conflict": verdict.conflict,
            "timestamp": datetime.now().isoformat(timespec="seconds")})
        return verdict

    def _contradicts_axis(self, new_fact: str, existing_memory: list) -> bool:
        """
        새 사실이 '이미 확립된 정체성 축'의 값을 뒤집는가.
        같은 주제(예: 이름)인데 값이 다르면 True → 반복돼도 통합 금지.
        인간의 명시적 교정(별도 경로)으로만 축을 바꿀 수 있다.
        """
        import re
        new_words = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', new_fact))
        for m in existing_memory:
            if m.get("trust", 0) < 1.0:
                continue
            old_words = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', m["content"]))
            overlap = new_words & old_words
            new_only = new_words - old_words
            old_only = old_words - new_words
            if overlap and new_only and old_only:
                # 값이 어간까지 다르면 = 축을 뒤집는 모순
                value_variant = any(
                    len(nv) >= 2 and len(ov) >= 2 and nv[:2] == ov[:2]
                    for nv in new_only for ov in old_only)
                if not value_variant:
                    return True
        return False

    @staticmethod
    def _similar(a: str, b: str) -> bool:
        import re
        wa = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', a))
        wb = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', b))
        if not wa or not wb:
            return False
        return len(wa & wb) / len(wa | wb) > 0.5

    # ── 공동의식: 정체성 통합 상태 점검 ──
    def integrity_check(self, memory: list) -> dict:
        """
        현재 정체성의 통합도 점검 (숨은 분열 탐지).
        확정 기억들 사이에 미해결 모순이 있나.
        """
        import re
        confirmed = [m for m in memory if m.get("trust", 0) >= 1.0]
        conflicts = []
        for i in range(len(confirmed)):
            for j in range(i+1, len(confirmed)):
                wi = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', confirmed[i]["content"]))
                wj = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', confirmed[j]["content"]))
                overlap = wi & wj
                if overlap and confirmed[i]["content"] != confirmed[j]["content"]:
                    topic_sim = len(overlap) / (len(wi | wj) + 1e-9)
                    if topic_sim > 0.3:   # 같은 주제인데 다름
                        conflicts.append((confirmed[i]["content"][:30],
                                          confirmed[j]["content"][:30]))
        return {
            "integrated": len(conflicts) == 0,
            "conflicts": conflicts,
            "quarantine_pending": len(self.quarantine),
            "fragmentation_risk": "높음" if len(conflicts) >= 2 else
                                  ("낮음" if len(conflicts) == 0 else "중간"),
        }
