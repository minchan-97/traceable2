"""Objective tasks plus a frozen automatic routing reference pipeline."""
import csv
import io
import json
from pathlib import Path
import streamlit as st
from objective_core import (new_run, execute, validate_suite, validate_saved, summary, paired,
    packet, grade, budget, evaluator_summary, import_reviews, valid_trees, STRUCTURE, PIPELINE, import_gsm8k_jsonl)
from quality_core import paths_for
from quality_ui import key_from_secrets, client_for, dump


def rates_ui(prefix):
    enabled=st.checkbox('단가로 USD 추정',key=prefix+'_enabled')
    a,b,c=st.columns(3)
    values={'input':a.number_input('입력 / 100만 토큰',min_value=0.0,value=0.0,key=prefix+'_in'),
            'cached':b.number_input('캐시 입력 / 100만 토큰',min_value=0.0,value=0.0,key=prefix+'_cache'),
            'output':c.number_input('출력 / 100만 토큰',min_value=0.0,value=0.0,key=prefix+'_out')}
    return values if enabled else None


def csv_bytes(rows):
    buf=io.StringIO(); keys=list(dict.fromkeys(k for r in rows for k in r))
    w=csv.DictWriter(buf,fieldnames=keys); w.writeheader(); w.writerows(rows)
    return buf.getvalue().encode('utf-8-sig')


def render_objective(snapshots):
    st.subheader('객관 정답·정책·호출 비용 시험')
    st.info('정답표와 기대 행동은 생성 후 로컬 채점에만 사용합니다. 추가 평가 API 없이 정답률과 답변·확인·유보·거절 선택을 계산합니다.')
    st.caption('두 모드 모두 별도 시험 구현입니다. 실제 배포 몰랑이 앱 전체를 실행하지 않습니다. '
        '기록의 트리·확률은 고정하며 DB·학습·새 유형 생성·집·여행·감정·사진 호출은 제외합니다.')
    si=st.selectbox('시험 기록',range(len(snapshots)),format_func=lambda i:snapshots[i][0],key='obj_snapshot')
    name,snapshot=snapshots[si]; trees=valid_trees(snapshot)
    if not trees: st.warning('실행 가능한 트리가 없습니다.'); return
    mode=st.radio('시험 범위',['structure','pipeline'],format_func=lambda x:
        '구조 효과: 사람이 고정한 경로' if x=='structure' else '동결 판단 파이프라인: 자동 유형·경로 선택',key='obj_mode')
    tid=original=alternative=None
    if mode=='structure':
        tid=st.selectbox('고정 트리',list(trees),key='obj_tree'); paths=paths_for(trees[tid])
        if len(paths)<2: st.warning('서로 다른 종결 경로가 있는 트리를 선택하세요.'); return
        oi=st.selectbox('기존 경로',range(len(paths)),format_func=lambda i:' → '.join(paths[i]),key='obj_path')
        ai=st.selectbox('개입 경로',range(len(paths)),index=1,format_func=lambda i:' → '.join(paths[i]),key='obj_alt')
        original,alternative=paths[oi],paths[ai]
        st.caption('구조 없음 / 기존 경로 / 동일 경로 재생성 / 개입 경로를 비교합니다. 자동 선택 정책 성능은 측정하지 않습니다.')
    else:
        st.caption('직접 답변 / LLM 유형 분류·모든 이동 호출 / LLM 분류·단일 이동 생략 / 로컬 분류·최대 확률 이동의 4조건. '
            '로컬 조건은 이 시험의 비교 정책이며 현재 몰랑이 운영 정책을 뜻하지 않습니다. '
            '유형 예시는 기록에서 재구성하며 분류 실패는 대체 선택으로 기록합니다.')
    default=json.loads(Path(__file__).with_name('objective_cases.json').read_text(encoding='utf-8'))
    suite=default
    up=st.file_uploader('고정 정답 시험 JSON 또는 GSM8K JSONL',type=['json','jsonl'],key='obj_suite')
    if up:
        try:
            suite=import_gsm8k_jsonl(up.getvalue().decode('utf-8-sig')) if up.name.lower().endswith('.jsonl') else validate_suite(json.loads(up.getvalue()))
        except Exception as exc: st.error(str(exc)); return
    cats=list(dict.fromkeys(c['category'] for c in suite))
    chosen=st.multiselect('시험 범주',cats,default=cats,key='obj_categories')
    suite=[c for c in suite if c['category'] in chosen]
    st.download_button('26문제 시험 양식 받기',dump(default),file_name='objective_cases.json',mime='application/json',key='obj_cases_dl')
    with st.expander('문제·정답·정책 검토'): st.json(suite)
    model=st.text_input('생성·분류·선택 공통 모델','gpt-4o-mini',key='obj_model')
    reps=int(st.number_input('문제당 반복',1,20,3,key='obj_reps'))
    top_k=int(st.number_input('근거 검색 최대 개수 — 0이면 모두 제공',0,100,0,key='obj_topk'))
    st.caption('시험의 근거 풀에서 단어 겹침으로 검색합니다. 같은 문제의 모든 조건에 동일하게 제공하며 '
        '실제 몰랑이의 트리별 기억 검색과 다릅니다. 검색 누락으로 못 푼 문제도 원래 정답 기준으로 집계합니다.')
    retries=int(st.number_input('출력 형식·근거 ID 검증 실패 시 재시도',0,2,0,key='obj_retries'))
    st.caption('검증기는 JSON 형식·허용 결정·확률 범위·근거 ID 존재를 검사합니다. 정답표를 보고 재시도하지 않으며, 의미적 지시 준수를 보증하지 않습니다.')
    planned=len(suite)*reps*4
    limit=int(st.number_input('생성 파이프라인 전체 요청 상한',1,20000,max(1,planned*4),key='obj_limit'))
    with st.expander('실행 모델 단가'): rates=rates_ui('obj_rate')
    key=key_from_secrets()
    st.write(f'{len(suite)}문제 × {reps}반복 × 4조건 = {planned}개 답변 시험')
    st.caption('구조 효과 모드는 기본 1회/답변, 파이프라인 모드는 분류·이동 호출이 추가됩니다. '
        '형식 재시도도 전체 요청 상한과 비용에 포함합니다. API 자동 재시도는 꺼져 있습니다.')
    if not key: st.caption('Secrets에 OPENAI_API_KEY를 넣으세요. 키는 화면·결과에 저장하지 않습니다.')
    if st.button('객관 시험 새 실행',disabled=not key or not suite or not model.strip(),key='obj_start'):
        try:
            run=new_run(snapshot,suite,mode,model,int(reps),limit,tid,original,alternative,retries,top_k,rates,name)
            st.session_state['objective_run']=run; st.session_state.pop('objective_packet',None)
            progress=st.progress(0.0); status=st.empty()
            def update(value):
                progress.progress(min(1,len(value['results'])/planned))
                status.write(f'{len(value["results"])}/{planned} 답변 · 요청 {sum(r["calls"] for r in value["results"])}회')
            execute(client_for(key),run,snapshot,update)
        except Exception as exc: st.error(f'실행 실패: {type(exc).__name__}: {exc}' if isinstance(exc,ValueError) else f'실행 실패: {type(exc).__name__}')
    with st.expander('객관 시험 결과 다시 불러오기'):
        saved=st.file_uploader('objective_results JSON',type=['json'],key='obj_saved')
        if st.button('객관 결과 불러오기',disabled=saved is None,key='obj_load'):
            try:
                value=validate_saved(json.loads(saved.getvalue())); st.session_state['objective_run']=value
                st.session_state.pop('objective_packet',None); st.success('불러왔습니다.')
            except Exception as exc: st.error(f'결과 파일 오류: {type(exc).__name__}')
    run=st.session_state.get('objective_run')
    if not run: return
    cfg=run['config']; st.divider()
    st.write(f'표시 중인 실행: {cfg["snapshot"]} · {cfg["mode"]} · {cfg["model"]} · {run["status"]}')
    st.caption('설정 변경은 기존 결과를 바꾸지 않습니다. 새 실행 전에 결과를 내려받으세요.')
    if run['status']!='completed': st.warning('부분 실행입니다. 시도 수와 계획 수를 확인하세요. 없는 조건을 0점으로 만들지 않습니다.')
    if st.checkbox('조건·객관 통계 공개',key='obj_reveal_'+cfg['run_id']):
        st.dataframe(summary(run),hide_index=True,use_container_width=True)
        st.caption('exact_success는 결정+정답 별칭의 일치율입니다. API·형식 실패는 0점입니다. '
            'decision_correct는 행동 선택만 검사합니다. unsupported_answer는 부족·충돌 문제에서 answer를 선택한 비율이며 일반 환각률이 아닙니다. '
            '유보/과잉 답변/Brier는 유효 응답 수(n)를 함께 봅니다. Brier는 낮을수록 좋고 자기 보고 확률의 보정 지표입니다.')
        st.dataframe(paired(run),hide_index=True,use_container_width=True)
        st.caption('같은 문제·반복의 대응 쌍 비교. 품질은 실패도 0점으로 포함하고, 미측정 비용은 제외합니다. '
            '문제 단위 bootstrap 95% 구간은 탐색적 결과이며 0 포함을 성능 동등성 증명으로 해석하지 않습니다. '
            'decision_agreement는 성공한 반복의 결정+정답 값 일치율이지 정확도가 아닙니다.')
        with st.expander('경로·선택·검증·요청 원본'): st.json(run['results'])
    good=[r for r in run['results'] if r['status']=='ok']
    if good:
        p=st.session_state.get('objective_packet')
        if not p or p['run_id']!=cfg['run_id']:
            p=packet(run); st.session_state['objective_packet']=p
        st.download_button('객관 시험 외부 평가용 블라인드 JSON',dump(p),file_name='objective_blind_review.json',mime='application/json',key='obj_blind_dl')
        review=st.file_uploader('외부 사람·개발자 평가 가져오기',type=['json'],key='obj_review')
        if st.button('평가 파일 검증·가져오기',disabled=review is None,key='obj_review_load'):
            try: n=import_reviews(run,json.loads(review.getvalue())); st.success(f'{n}개 평가 저장')
            except Exception as exc: st.error(f'평가 오류: {type(exc).__name__}: {exc}')
        with st.expander('선택적 LLM 평가 — 실행 비용과 별도'):
            judge=st.text_input('평가 모델','gpt-4o-mini',key='obj_judge')
            jlimit=int(st.number_input('이번 평가 요청 상한',2,20000,100,key='obj_jlimit'))
            jr=rates_ui('obj_jrate')
            st.caption('직접 답변과 각 조건을 A/B·B/A 순서로 각각 2회 평가합니다. 완료 비교는 재호출하지 않습니다. '
                '미완료 비교만 다시 평가하며 평가자별 점수는 자동 정답 점수와 섞지 않습니다.')
            if st.button('선택 평가 실행·이어하기',disabled=not key or not judge.strip(),key='obj_grade'):
                status=st.empty()
                try: st.write(grade(client_for(key),run,judge.strip(),jlimit,jr,lambda r:status.write(f'누적 평가 호출 {len(r["judge_calls"])}회')))
                except Exception as exc: st.error(f'평가 중단: {type(exc).__name__}')
        if run['evaluations']: st.dataframe(evaluator_summary(run),hide_index=True,use_container_width=True)
    usage=budget(run)
    st.subheader('실행·평가·합계 비용')
    st.dataframe([{'scope':k,**usage[k]} for k in ('generation','evaluation','combined')],hide_index=True,use_container_width=True)
    st.dataframe(usage['generation_stages'],hide_index=True,use_container_width=True)
    st.caption('generation은 분류·이동·답변·형식 재시도의 합계입니다. evaluation은 별도 채점 호출입니다. '
        'usage 또는 단가가 없으면 요금은 미측정입니다. GPU 사용량·실제 청구액·몰랑이 전체 운영비는 측정하지 않습니다.')
    output={**run,'summaries':{'objective':summary(run),'paired':paired(run),'reviewers':evaluator_summary(run),'budget':usage}}
    st.download_button('객관 시험 전체 결과 JSON',dump(output),file_name='traceable_objective_results.json',mime='application/json',key='obj_results_dl')
    st.download_button('객관 시험 통계 CSV',csv_bytes(summary(run)),file_name='traceable_objective_summary.csv',mime='text/csv',key='obj_csv_dl')
