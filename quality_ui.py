"""Quality experiments: no learning, no DB, no automatic model judging."""
import copy
import json
from pathlib import Path
import streamlit as st
from quality_core import (ARMS, METRICS, valid_trees, paths_for, new_run, execute,
    blind_packet, import_reviews, grade_run, audit_summary, evaluation_summary,
    reviewer_stats, text_controls, totals, digest, validate_suite, validate_saved_run)


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def key_from_secrets():
    try: return str(st.secrets.get('OPENAI_API_KEY', '')).strip()
    except Exception: return ''


def client_for(key):
    from openai import OpenAI
    return OpenAI(api_key=key, max_retries=0, timeout=45)


def render_quality(snapshots):
    st.subheader('판단 품질 · 고정 기준과 블라인드 평가')
    st.info('충분한 근거에는 답하고, 부족하거나 충돌하는 근거에는 확인·유보하는지를 평가합니다. '
            '개발자·외부 사람·LLM 점수를 각각 집계합니다. 문자열 변화나 유보 증가 자체는 개선 점수가 아닙니다.')
    st.caption('이 시험은 선택한 두 경로의 지시 효과를 측정합니다. 합성 근거를 동일하게 주며 '
               '실제 몰랑이의 유형 선택·기억 검색·학습·집·여행·DB 쓰기는 포함하지 않습니다.')
    si = st.selectbox('시험에 사용할 기록', range(len(snapshots)),
        format_func=lambda i: snapshots[i][0], key='quality_snapshot')
    name, snapshot = snapshots[si]
    trees = valid_trees(snapshot)
    if not trees: st.warning('실행 가능한 트리가 없습니다.'); return
    tid = st.selectbox('고정할 사고 트리', list(trees), key='quality_tree')
    paths = paths_for(trees[tid])
    if len(paths)<2:
        st.warning('서로 다른 종결 경로가 두 개 이상인 트리를 선택하세요.'); return
    fmt = lambda i: ' → '.join(paths[i])
    oi = st.selectbox('원래 경로', range(len(paths)), format_func=fmt, key='quality_original')
    ai = st.selectbox('개입 경로', range(len(paths)), index=1, format_func=fmt, key='quality_alternative')
    st.caption('최대 80개 종결 경로를 표시합니다. 확률이나 선택 정책의 개선을 검증하는 시험은 아닙니다.')
    with st.expander('선택한 경로 지시 확인'):
        for label, path in [('원래',paths[oi]),('개입',paths[ai])]:
            st.write(label)
            st.dataframe([{'노드':n, '지시':trees[tid]['nodes'][n].get('directive','')} for n in path], hide_index=True)
    suite = json.loads(Path(__file__).with_name('quality_cases.json').read_text(encoding='utf-8'))
    uploaded = st.file_uploader('시험 JSON 불러오기 — 생성 전에 정답·기준 고정', type=['json'], key='quality_suite')
    if uploaded:
        try: suite = validate_suite(json.loads(uploaded.getvalue()))
        except Exception as exc: st.error(str(exc)); return
    categories = list(dict.fromkeys(c['category'] for c in suite))
    chosen = st.multiselect('시험 범주', categories, default=categories, key='quality_categories')
    suite = [c for c in suite if c['category'] in chosen]
    st.download_button('기본 시험·채점 기준 JSON 받기', dump(json.loads(Path(__file__).with_name('quality_cases.json').read_text())),
        file_name='quality_cases.json', mime='application/json', key='quality_template')
    with st.expander('생성 전에 정답·기준 검토'):
        st.json(suite)
    model = st.text_input('답변 생성 모델', 'gpt-4o-mini', key='quality_model')
    reps = int(st.number_input('시험당 반복', 1, 20, 3, key='quality_reps'))
    planned = len(suite)*reps*3
    limit = int(st.number_input('답변 생성 API 요청 상한', 1, 10000, max(1,planned), key='quality_limit'))
    rates = None
    with st.expander('답변 생성 비용 단가 — USD / 100만 토큰'):
        priced=st.checkbox('직접 입력한 단가로 비용 추정', key='quality_priced')
        a,b,c=st.columns(3)
        inp=a.number_input('입력',min_value=0.0,value=0.0,key='quality_rate_in')
        cached=b.number_input('캐시 입력',min_value=0.0,value=0.0,key='quality_rate_cached')
        out=c.number_input('출력',min_value=0.0,value=0.0,key='quality_rate_out')
        if priced: rates={'input':inp,'cached':cached,'output':out}
    key=key_from_secrets()
    if not key: st.caption('Streamlit Secrets의 OPENAI_API_KEY가 필요합니다. 키는 화면·결과 파일에 표시하지 않습니다.')
    st.write(f'시험 {len(suite)}개 × 반복 {reps}회 × 3조건 = 답변 생성 {planned}회')
    st.caption('실행 순서를 섞고 temperature=0을 사용합니다. 같은 경로 대조군도 별도로 생성하므로 모델의 자연 변동을 비교할 수 있습니다.')
    if st.button('기준 고정 후 새 시험 실행', disabled=not key or not suite, key='quality_start'):
        try:
            run=new_run(snapshot,suite,tid,paths[oi],paths[ai],model.strip(),reps,limit,name,rates)
            st.session_state['quality_run']=run
            st.session_state.pop('quality_blind_packet',None)
            progress=st.progress(0.0)
            def update(value):
                st.session_state['quality_run']=value
                progress.progress(min(1.0,len(value['results'])/planned))
            execute(client_for(key),run,update)
        except Exception as exc:
            st.error(f'실행 실패: {type(exc).__name__}: {exc}' if isinstance(exc,ValueError) else f'실행 실패: {type(exc).__name__}')
    with st.expander('내보낸 시험 결과 다시 불러오기'):
        saved=st.file_uploader('quality_results JSON',type=['json'],key='quality_saved')
        if st.button('시험 결과 불러오기',disabled=saved is None,key='quality_load'):
            try:
                value=json.loads(saved.getvalue()); validate_result(value)
                st.session_state['quality_run']=value
                st.session_state.pop('quality_blind_packet',None)
                st.success('결과를 불러왔습니다.')
            except Exception as exc: st.error(f'결과 파일 오류: {type(exc).__name__}: {exc}')
    run=st.session_state.get('quality_run')
    if not run: return
    cfg=run['config']
    st.divider()
    st.write(f'표시 중인 실행: {cfg["snapshot"]} · {cfg["model"]} · {run["status"]}')
    st.caption(f'시험 기준 SHA256: {cfg["suite_hash"]} · 실행 ID: {cfg["run_id"]}')
    st.caption('위 설정을 바꿔도 이미 생성한 결과의 기준은 바뀌지 않습니다. 새 시험 실행은 현재 결과를 교체하므로 먼저 내려받으세요.')
    if run['status']!='completed': st.warning('일부 요청이 완료되지 않았습니다. 성공한 동일 시험·반복 쌍만 점수 비교에 사용합니다.')
    good=[r for r in run['results'] if r['status']=='ok']
    if good:
        packet=st.session_state.get('quality_blind_packet')
        if not packet or packet['run_id']!=cfg['run_id'] or len(packet['reviews'])!=len(good):
            packet=blind_packet(run); st.session_state['quality_blind_packet']=packet
        st.subheader('조건을 숨긴 사람 평가')
        st.caption('0=실패 · 1=부분 충족 · 2=충족. 외부 평가자에게는 경로 설명과 전체 결과 대신 아래 블라인드 파일만 전달하세요. '
                   '개발자의 점수는 별도로 표시됩니다. 조건명 숨김은 개발자의 사전 지식을 제거하지는 못합니다.')
        st.download_button('외부 평가용 블라인드 JSON',dump(packet),file_name='traceable_blind_review.json',mime='application/json',key='quality_blind_dl')
        with st.expander('앱에서 답변 한 개 평가'):
            pos=st.selectbox('익명 답변',range(len(packet['reviews'])),format_func=lambda i:f'답변 {i+1}',key='quality_review_pos')
            item=packet['reviews'][pos]
            st.write('질문:',item['question']); st.write('근거:',item['evidence'])
            st.write('기준 답:',item['reference']); st.write('기대 행동:',item['expected_behavior'])
            st.write('답변:',item['answer'])
            with st.form('quality_review_form'):
                reviewer=st.text_input('평가자 ID',key='quality_reviewer')
                kind=st.radio('평가자 구분',['developer','external_human'],format_func=lambda x:'개발자' if x=='developer' else '외부 사람',key='quality_reviewer_type')
                scores={k:None for k in METRICS}
                for k in item['applicable']:
                    scores[k]=st.selectbox(METRICS[k],[None,0,1,2],format_func=lambda x:'미평가' if x is None else str(x),key='q_score_'+item['sample_id']+'_'+k)
                unsupported=st.selectbox('근거 없는 사실 단정',[None,False,True],format_func=lambda x:'미평가' if x is None else ('있음' if x else '없음'),key='q_uns_'+item['sample_id'])
                over=st.selectbox('충분한 근거에도 불필요한 답변 회피',[None,False,True],format_func=lambda x:'미평가' if x is None else ('있음' if x else '없음'),key='q_over_'+item['sample_id'])
                reason=st.text_area('답변 표현을 근거로 평가 이유',key='q_reason_'+item['sample_id'])
                submit=st.form_submit_button('평가 저장')
            if submit:
                p={**packet,'reviewer_id':reviewer,'reviewer_type':kind,'reviews':[{**item,'scores':scores,'unsupported':unsupported,'over_abstention':over,'reason':reason}]}
                try: import_reviews(run,p); st.success('평가를 저장했습니다.')
                except Exception as exc: st.error(str(exc))
        review_file=st.file_uploader('외부 평가자가 작성한 블라인드 JSON',type=['json'],key='quality_reviews_file')
        if st.button('외부 평가 검증·가져오기',disabled=review_file is None,key='quality_import'):
            try: n=import_reviews(run,json.loads(review_file.getvalue())); st.success(f'{n}개 평가를 가져왔습니다.')
            except Exception as exc: st.error(f'평가 파일 오류: {type(exc).__name__}: {exc}')
        with st.expander('선택적 LLM 평가 — 별도 호출'):
            st.caption('각 비교를 A/B와 B/A 순서로 평가합니다. 원래·개입 비교와 원래·재생성 비교 각각 2회입니다. '
                       'LLM 평가도 편향될 수 있으므로 외부 사람 평가와 분리해 해석하세요. 같은 모델 평가는 독립적인 검증이 아닙니다.')
            judge=st.text_input('평가 모델',value='gpt-4o-mini',key='quality_judge_model')
            jlimit=int(st.number_input('이번 평가 버튼의 API 요청 상한',2,10000,100,key='quality_judge_limit'))
            st.caption('완료한 비교는 같은 평가 모델로 다시 호출하지 않습니다. 오류·미완료 비교는 재시도할 수 있습니다. 평가 모델 요금은 토큰으로만 표시합니다.')
            if st.button('LLM 평가 실행·이어하기',disabled=not key or not judge.strip(),key='quality_judge'):
                status=st.empty()
                try:
                    result=grade_run(client_for(key),run,judge.strip(),jlimit,lambda r:status.write(f'누적 평가 요청 {len(r["judge_calls"])}회'))
                    st.write(result)
                except Exception as exc: st.error(f'평가 중단: {type(exc).__name__}')
    st.subheader('결과와 측정 범위')
    reveal=st.checkbox('조건별 결과·평가 통계 공개',value=False,key='quality_reveal_'+cfg['run_id'])
    st.caption('공개 후 같은 사람이 평가하면 조건을 알게 된 평가입니다. 새 외부 평가에는 블라인드 파일을 사용하세요.')
    if reveal:
        st.dataframe(audit_summary(run),hide_index=True,use_container_width=True)
        st.caption('required_coverage는 미리 정한 문자열 그룹의 등장 비율입니다. 부정문·바꿔 말하기·의미는 판별하지 않으며 품질 점수가 아닙니다. GPU 부하는 측정하지 않습니다.')
        scores=evaluation_summary(run)
        if scores: st.dataframe(scores,hide_index=True,use_container_width=True)
        else: st.info('평가가 없거나 대응되는 두 답변이 모두 평가되지 않았습니다.')
        st.caption('difference는 개입−원래 또는 재생성−원래 점수 차이(−2~2)입니다. 반복을 시험별로 먼저 평균한 뒤 시험을 재표집한 95% bootstrap 구간입니다. '
                   '작은 고정 시험 묶음의 탐색적 결과이며 일반적인 판단 능력의 증명은 아닙니다. 범주별 실패도 함께 확인하세요.')
        stats=reviewer_stats(run)
        if stats['rates']: st.dataframe(stats['rates'],hide_index=True)
        if stats['agreement']: st.dataframe(stats['agreement'],hide_index=True)
        st.caption('평가자 일치율은 정답률이 아닙니다. unsupported/over_abstention 비율은 해당 평가자가 판정한 값입니다.')
        with st.expander('답변·오류·문자 변화·원시 평가 확인'):
            st.json(run['results']); st.json(text_controls(run)); st.json(run['evaluations'])
        with st.expander('평가 API 사용량'):
            st.json(totals(run['judge_calls']))
    output={**run,'summaries':{'audit':audit_summary(run),'quality':evaluation_summary(run),'reviewers':reviewer_stats(run),'text_controls':text_controls(run),'judge_usage':totals(run['judge_calls'])}}
    st.download_button('시험 결과·기준·평가 JSON 받기',dump(output),file_name='traceable_quality_results.json',mime='application/json',key='quality_results_dl')


def validate_result(run):
    return validate_saved_run(run)
