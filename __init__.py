"""
traceable — 판단 추적 가능성과 계획 실행을 재는 여덟 축.

표준 라이브러리만 쓴다. 모델도 GPU 도 필요 없다.
어떤 시스템이든 Trace 형식으로 기록만 남기면 같은 자로 잴 수 있다.

    from traceable import Trace, score, series, table

    t = Trace(units=[...], items=[...], state={...}, step=100, size=170)
    print(score(t))
"""
from .core import (Trace, score, series, table, report, load, save,
                   AXES, AXIS_KO)

__all__ = ["Trace", "score", "series", "table", "report", "load", "save",
           "AXES", "AXIS_KO"]
__version__ = "0.2.0"
