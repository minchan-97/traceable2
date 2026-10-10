# Traceable Streamlit

## 실행
압축을 풀고 traceable_streamlit 폴더에서 실행하세요.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

성장 기록 딕셔너리 피클 여러 개를 업로드할 수 있습니다. 모델·API 키·Supabase 접속은 필요 없습니다.
시점별 여덟 지표와 기록된 분기에 대한 대안 선택 실험을 제공합니다.
원본 피클은 수정하지 않습니다. 객체를 실행하는 피클은 거부합니다.
결과를 JSON으로 다운로드할 수 있습니다.

## 해석 범위
원래 경로는 기록 선택의 재생입니다. 역사적 LLM 판단의 독립 재현은 아닙니다.
대안 경로는 현재 스냅샷의 그래프를 사용하고 학습 전 확률은 반올림된 당시 기록을 사용합니다.
실제 LLM 발화는 생성하지 않습니다. 최종 답변 변화나 품질 개선을 입증하지 않습니다.
Traceable 집계 점수의 제약은 README_experiment.md를 참고하세요.

## 검증
Python 구문 검사 통과. 81번의 분기 개입 149건 앱 계산 함수 검사 통과.
객체 피클 거부 검사 통과. Traceable 기존 테스트 13개 통과.
개발 환경에 Streamlit이 설치되어 있지 않아 화면 실행 검증은 하지 못했습니다.
