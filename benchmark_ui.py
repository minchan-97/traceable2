import csv
import io
import json
import random
import hashlib
import streamlit as st
from benchmark_core import ARMS, valid_trees, run_batch, summary, paired_summary

def render_benchmark(snapshots):
    st.subheader('같은 질문으로 세 조건 비교')
    st.caption('피클을 고정하고 학습·DB 쓰기 없이 실행합니다. GPU 사용량은 측정하지 않습니다.')
    st.info('이 실험은 문자 판단 호출을 비교하는 통제 실험입니다. 집·여행 맥락 주입과 새 트리 생성, '
            '사진·표정 생성은 포함하지 않습니다. 기억은 세 조건에 동일하게 제공하며, 실제 앱 전체 비용으로 해석하지 않습니다.')
    si=st.selectbox('벤치마크 피클',range(len(snapshots)),format_func=lambda i:snapshots[i][0],key='bench_snapshot')
    name,snapshot=snapshots[si]
    trees=valid_trees(snapshot)
    if not trees:
        st.error('실행 가능한 트리가 없습니다.'); return
    questions_raw=st.text_area('질문 묶음 — 한 줄에 한 질문',
        '오늘 좀 지쳤어.\n내가 좋아하는 노래 기억나?\n근거가 부족한 소문을 어떻게 판단할까?',height=140)
    qfile=st.file_uploader('질문 TXT 불러오기',type=['txt'],key='bench_questions_file')
    if qfile: questions_raw=qfile.getvalue().decode('utf-8-sig')
    questions=[q.strip() for q in questions_raw.splitlines() if q.strip()]
    model=st.text_input('세 조건 공통 모델',value='gpt-4o-mini',key='bench_model')
    reps=st.number_input('질문당 반복 횟수',1,20,3,key='bench_reps')
    mode=st.radio('유형 선택', ['LLM 분류 포함', '트리 고정'],horizontal=True)
    tid=st.selectbox('고정할 트리',list(trees),disabled=mode!='트리 고정',key='bench_tree')
    st.caption('피클에는 원래 type_examples가 없어 유형 목록 예시를 기록에서 재구성합니다. '
               '분류 실패 시 가장 많이 사용한 트리로 넘어가며 결과에 표시됩니다. '
               '새 유형 생성 비용은 이 비교에 포함하지 않습니다.')
    auxiliary=st.checkbox('세 조건 모두 감정·기억 추출 호출 추가',value=False)
    st.caption('보조 호출 포함 시 각 조건에 2회가 더해집니다. 추출 결과는 저장하지 않습니다.')
    extra=st.text_area('세 조건에 동일하게 줄 추가 배경',key='bench_extra')
    limit=st.number_input('이 실행의 API 요청 상한',min_value=1,max_value=10000,value=300,step=10)
    rates=None
    with st.expander('요금 추정 — 직접 입력한 단가로 계산'):
        priced=st.checkbox('USD / 100만 토큰 단가 사용',value=False)
        a,b,c=st.columns(3)
        inp=a.number_input('입력 단가',0.0,value=0.0,format='%.4f')
        cached=b.number_input('캐시 입력 단가',0.0,value=0.0,format='%.4f')
        out=c.number_input('출력 단가',0.0,value=0.0,format='%.4f')
        st.caption('모델 요금표의 실제 단가를 입력하세요. 단가 미입력 시 토큰만 비교합니다. '
                   'usage가 없는 응답은 요금을 미측정으로 남깁니다.')
        if priced: rates={'input':inp,'cached':cached,'output':out}
    try: key=str(st.secrets.get('OPENAI_API_KEY','')).strip()
    except Exception: key=''
    if not key: st.info('Streamlit Secrets에 OPENAI_API_KEY를 넣으면 실행할 수 있습니다.')
    planned=len(questions)*int(reps)*3
    st.write(f'질문 {len(questions)}개 × 반복 {reps}회 × 조건 3개 = {planned}개 답변 실험')
    st.caption('B는 모든 이동에서 모델을 호출하고 C는 선택지가 하나인 이동을 생략합니다. '
               '모델 출력은 완전히 결정적이지 않으므로 질문별 조건 실행 순서를 섞습니다.')
    if st.button('벤치마크 실행',disabled=not key or not questions or not model.strip()):
        from openai import OpenAI
        config={'snapshot':name,'model':model.strip(),'repetitions':int(reps),'questions':questions,
                'fixed_tree':tid if mode=='트리 고정' else None,'auxiliary':auxiliary,
                'rates':rates,'limit':int(limit),'extra_context':extra,
                'snapshot_sha256':hashlib.sha256(json.dumps(snapshot,ensure_ascii=False,sort_keys=True,default=str).encode()).hexdigest(),
                'protocol':'frozen-shared-memory-v1','classify_temperature':0,'choose_temperature':.3,'answer_temperature':.7}
        st.session_state['bench_results']=[]
        st.session_state['bench_config']=config
        st.session_state['bench_ratings']={}
        st.session_state['bench_blind_order']=None
        progress=st.progress(0.0); status=st.empty()
        def update(results):
            st.session_state['bench_results']=list(results)
            progress.progress(min(1.0,len(results)/planned))
            status.write(f'{len(results)}/{planned} 완료 · API {sum(r["calls"] for r in results)}회')
        try:
            run_batch(OpenAI(api_key=key,max_retries=0,timeout=45),snapshot,questions,
                model.strip(),int(reps),config['fixed_tree'],auxiliary,extra,int(limit),rates,update)
        except Exception as exc:
            st.error(f'실행 중단: {type(exc).__name__}. 완료된 결과는 아래에 남아 있습니다.')
    results=st.session_state.get('bench_results',[])
    if not results: return
    config=st.session_state['bench_config']
    st.caption(f'표시 중인 실행: {config["snapshot"]} · {config["model"]} · {config["protocol"]}')
    st.dataframe(summary(results),hide_index=True,use_container_width=True)
    st.caption('exact_text_agreement는 같은 질문 반복 간 문자의 일치율입니다. 판단 안정성·정확도와 동일한 지표가 아닙니다.')
    if len(results)<len(config['questions'])*config['repetitions']*3:
        st.warning('일부 조건이 완료되지 않았습니다. 같은 질문·반복의 세 조건이 모두 끝난 묶음끼리 비교하세요.')
    st.subheader('조건명을 가리고 답변 평가')
    ratings=st.session_state.setdefault('bench_ratings',{})
    order=st.session_state.get('bench_blind_order')
    if order is None:
        order=list(range(len(results))); random.SystemRandom().shuffle(order)
        st.session_state['bench_blind_order']=order
    pos=st.selectbox('평가할 답변',range(len(order)),format_func=lambda i:f'답변 {i+1}',key='bench_review')
    row=results[order[pos]]
    st.write('질문:',row['question']); st.write(row.get('answer',''))
    if row['status']!='ok':st.warning('API 오류 또는 호출 상한으로 완료되지 않은 답변입니다.')
    with st.expander('동일하게 제공한 근거'):
        st.write(row['evidence'])
    sid=row['sample_id']; existing=ratings.get(sid,{})
    with st.form('bench_rating_'+sid):
        values={}
        for field,label in [('relevance','질문에 답했는가'),('correctness','사실·추론이 맞는가'),
                            ('uncertainty','근거 부족을 적절히 표현했는가')]:
            values[field]=st.selectbox(label,[-1,0,1,2],index=[-1,0,1,2].index(existing.get(field,-1)),
                                     format_func=lambda n:{-1:'미평가',0:'부족',1:'부분 충족',2:'충족'}[n])
        values['conclusion']=st.text_input('판단 결과 요약 — 반복 간 일관성 비교용',existing.get('conclusion',''))
        values['notes']=st.text_area('평가 근거',existing.get('notes',''))
        if st.form_submit_button('평가 저장'):
            ratings[sid]=values;st.success('저장했습니다.')
    if st.checkbox('조건과 경로 공개',key='bench_unblind'):
        st.write(ARMS[row['arm']],row.get('path',[]))
        aggregates=[]
        for arm in ARMS:
            rr=[ratings.get(r['sample_id'],{}) for r in results if r['arm']==arm]
            agg={'조건':ARMS[arm]}
            for field in ('relevance','correctness','uncertainty'):
                vals=[x[field] for x in rr if x.get(field,-1)>=0]
                agg[field+'_n']=len(vals);agg[field]=sum(vals)/len(vals) if vals else None
            aggregates.append(agg)
        st.dataframe(aggregates,hide_index=True)
        st.dataframe(paired_summary(results,ratings),hide_index=True,use_container_width=True)
        st.caption('차이는 오른쪽 조건 − 왼쪽 조건입니다. 같은 질문·반복에서 성공한 쌍만 비교하고, '
                   '95% 구간은 질문 단위 재표집으로 계산합니다. 소수 질문·주관적 평가로 성능 동등성을 확정하지 마세요.')
        st.caption('평균 점수는 직접 평가한 항목만 집계합니다. 미평가·실패 건수와 표본 수를 함께 보세요.')
        with st.expander('호출별 사용량과 실패 기록'):
            st.json(row['call_log'])
    payload={'config':config,'results':results,'ratings':ratings,'summary':summary(results),
             'paired_summary':paired_summary(results,ratings)}
    st.download_button('벤치마크 JSON 받기',json.dumps(payload,ensure_ascii=False,indent=2),
                       file_name='traceable_benchmark.json',mime='application/json')
    flat=[]
    for r in results:
        flat.append({k:r.get(k) for k in ('sample_id','question_index','repetition','arm','question','status','answer',
                        'calls','input_tokens','output_tokens','cached_tokens','estimated_usd','wall_seconds',
                        'trace_bytes','trace_serialization_ms')}|
                    ratings.get(r['sample_id'],{}))
    buf=io.StringIO(); cols=list(dict.fromkeys(k for r in flat for k in r))
    w=csv.DictWriter(buf,fieldnames=cols);w.writeheader();w.writerows(flat)
    st.download_button('벤치마크 CSV 받기',buf.getvalue().encode('utf-8-sig'),
                       file_name='traceable_benchmark.csv',mime='text/csv')
