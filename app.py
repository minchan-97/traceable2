"""Run: streamlit run app.py"""
import copy
import hashlib
import io
import pickle
from dataclasses import fields
import streamlit as st
from pathlib import Path
import sys

# Streamlit 배포에서도 앱과 같은 폴더의 모듈을 먼저 사용한다.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from traceable_core import series, AXIS_KO
from traceable_adapters import from_molang
from thought_structure import ThoughtStructure, JudgmentNode
from answer_probe import prepare_pair, generate_pair
from benchmark_ui import render_benchmark
from quality_ui import render_quality
from objective_ui import render_objective
from benchmark_core import intervention_control
from longrun_ui import render_longrun
from identity_stress_ui import render_identity_stress
from scaffold_ui import render_scaffold

st.set_page_config(page_title='Traceable · 몰랑이', page_icon='🌳', layout='wide')
st.title('Traceable · 몰랑이 기록 분석')
st.caption('v6 · 시점 비교 · 장기 안정성 관찰 · 정체성 충격 시험 · 인간 비계 · 분기 개입 · 객관 정답과 동결 판단 파이프라인 시험')

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError('객체형 피클은 지원하지 않습니다. 성장 기록 딕셔너리를 올려주세요.')

def read_snapshot(raw):
    data = SafeUnpickler(io.BytesIO(raw)).load()
    if not isinstance(data, dict) or not isinstance(data.get('trees'), dict):
        raise ValueError('trees가 있는 몰랑이 성장 기록이 필요합니다.')
    return data

def restore_tree(data):
    tree = ThoughtStructure(data.get('lr', .1), data.get('continuity', .8))
    tree.nodes = {key: JudgmentNode(**{k: v for k, v in node.items()
                    if k in {f.name for f in fields(JudgmentNode)}})
                  for key, node in data['nodes'].items()}
    tree.transitions = copy.deepcopy(data['transitions'])
    tree.root_id = data['root_id']
    return tree

def experiment(data, history, branch, alternative):
    tree = restore_tree(data)
    for step in history['at_probs']:
        tree.transitions[step['at']] = copy.deepcopy(step['options'])
    choices = {s['at']: s['chose'] for s in history['at_probs']}
    def pick(node, candidates, nodes, probs, context):
        return choices.get(node.id, max(zip(candidates, probs), key=lambda p: p[1])[0])
    def change(node, candidates, nodes, probs, context):
        return alternative if node.id == branch else pick(node, candidates, nodes, probs, context)
    original = tree.traverse(choose_fn=pick, context=history.get('context', ''))
    changed = tree.traverse(choose_fn=change, context=history.get('context', ''))
    feedback = history.get('feedback')
    output = {'original_path': original.path, 'alternative_path': changed.path,
              'matches_record': original.path == history['path'],
              'original_directives': [tree.nodes[k].directive for k in original.path],
              'alternative_directives': [tree.nodes[k].directive for k in changed.path],
              'before': tree.transitions[branch], 'feedback': feedback}
    if feedback is not None:
        a, b = copy.deepcopy(tree), copy.deepcopy(tree)
        a.learn(original, feedback); b.learn(changed, feedback)
        output.update(after_original=a.transitions[branch], after_alternative=b.transitions[branch])
    return output

files = st.file_uploader('파일 올리기 — 분석은 .pkl만 지원', accept_multiple_files=True)
snapshots = []
for uploaded in files:
    if Path(uploaded.name).suffix.lower() != '.pkl':
        st.warning(f'{uploaded.name}: .pkl 파일만 분석할 수 있어 건너뜁니다.')
        continue
    try:
        snapshots.append((uploaded.name, read_snapshot(uploaded.getvalue())))
    except Exception as exc:
        st.error(f'{uploaded.name}: {exc}')
if not snapshots:
    st.info('몰랑이 성장 기록을 올리면 분석이 시작됩니다.')
    st.stop()
snapshots.sort(key=lambda pair: int(pair[1].get('meta', {}).get('cycle', 0)))
tab_scores, tab_longrun, tab_stress, tab_scaffold, tab_intervene, tab_benchmark, tab_quality, tab_objective = st.tabs(['시점별 지표', '장기 안정성 관찰', '정체성 충격 시험', '인간 비계', '분기 개입 실험', '비용·품질 벤치마크', '판단 품질 시험', '객관·파이프라인 시험'])
with tab_scores:
    scores = series([from_molang(d, label=name) for name, d in snapshots])
    rows = [{'파일': s['label'], '회차': s['step'], '사실 수': s['size'],
             **{AXIS_KO[k]: s[k] for k in AXIS_KO}} for s in scores]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption('0~1 비율. 빈 값은 미측정입니다. 집계 단위는 판단 건수가 아니라 트리입니다.')
    st.warning('이 점수는 XAI의 인과적 충실도 점수가 아닙니다. 근거 연결률은 트리의 기억 보유, 학습 반영률은 현재 가중치의 편차, 상태 유래율은 단어 겹침을 측정합니다. 출처에는 기본값이 포함될 수 있습니다.')
    st.caption('현재 몰랑이 어댑터는 replayable을 설정하지 않아 재현 가능률이 0으로 나올 수 있습니다. 이는 재현 실패율을 뜻하지 않습니다. 연속성은 확신한 동일 사실 문장의 유지율입니다.')
    st.caption('주의: 이 연속성은 기억 창(예전 400건) 밀림을 소실로 셉니다. 창 밀림과 진짜 소실을 나눈 비교는 "장기 안정성 관찰" 탭을 보세요.')
with tab_longrun:
    render_longrun(snapshots)
with tab_stress:
    render_identity_stress(snapshots)
with tab_scaffold:
    render_scaffold(snapshots)
    with st.expander('원시 집계 보기'):
        st.json(scores)
with tab_intervene:
    index = st.selectbox('분석할 시점', range(len(snapshots)), format_func=lambda i: snapshots[i][0])
    name, snapshot = snapshots[index]
    trees = snapshot['trees']
    eligible = {tid: [(i, h) for i, h in enumerate(t.get('history', []))
                     if any(len(s.get('options', {})) > 1 for s in h.get('at_probs', []))]
                for tid, t in trees.items()}
    eligible = {k: v for k, v in eligible.items() if v}
    if not eligible:
        st.info('당시 갈림길 확률이 저장된 기록이 없습니다.')
    else:
        tid = st.selectbox('판단 유형', list(eligible))
        records = eligible[tid]
        ri = st.selectbox('판단 기록', range(len(records)), index=len(records)-1,
                         format_func=lambda i: f"#{records[i][0]} · {records[i][1].get('timestamp', '')}")
        _, history = records[ri]
        branches = [s for s in history['at_probs'] if len(s['options']) > 1]
        bi = st.selectbox('개입할 갈림길', range(len(branches)), format_func=lambda i: branches[i]['at'])
        branch = branches[bi]
        st.write('원래 선택:', branch['chose'])
        st.json(branch['options'])
        alternative = st.selectbox('대신 선택할 노드', [k for k in branch['options'] if k != branch['chose']])
        st.caption('같은 맥락·피드백을 유지하고 선택 하나만 바꿉니다. 원래 선택은 기록에서 재생하며 LLM의 당시 판단을 다시 계산하지 않습니다. 확률은 반올림된 기록, 다른 경로의 노드는 저장 시점 구조를 사용합니다.')
        with st.expander('당시 입력과 답변'):
            st.text(history.get('context', ''))
            st.write(history.get('answer', ''))
        try:
            result = experiment(trees[tid], history, branch['at'], alternative)
            st.write('원래 경로 재생 일치:', result['matches_record'])
            left, right = st.columns(2)
            with left:
                st.subheader('원래 선택')
                st.write(' → '.join(result['original_path']))
                for directive in result['original_directives']:
                    st.write(directive)
            with right:
                st.subheader('대안 선택')
                st.write(' → '.join(result['alternative_path']))
                for directive in result['alternative_directives']:
                    st.write(directive)
            if result['feedback'] is not None:
                st.write('동일하게 적용한 피드백:', result['feedback'])
                st.dataframe([{'다음 노드': k, '학습 전': v,
                               '원래 선택 후': result['after_original'].get(k),
                               '대안 선택 후': result['after_alternative'].get(k)}
                              for k, v in result['before'].items()], hide_index=True, use_container_width=True)
            else:
                st.info('저장된 피드백이 없어 학습 결과 비교는 생략했습니다.')
            st.divider()
            st.subheader('최종 답변 개입 실험')
            st.caption('몰랑이 답변 프롬프트 형식을 사용합니다. 저장된 context를 두 경로에 동일하게 넣습니다. '
                       '당시의 전체 기억·집·여행 프롬프트는 기록에 없을 수 있어 과거 응답의 완전한 재현은 아닙니다.')
            wiring = st.radio('답변 노드 연결',
                              ['최신 몰랑이: 판단 노드 연결', '이전 앱: 빈 노드 사전'],
                              horizontal=True, key='answer_wiring')
            node_map = {} if wiring.startswith('이전') else restore_tree(trees[tid]).nodes
            prepared = prepare_pair(result['original_path'], result['alternative_path'],
                                    history.get('context', ''), node_map)
            st.write('모델에 전달되는 프롬프트가 달라지는가:', prepared['prompt_changed'])
            if not prepared['prompt_changed']:
                st.info('이 연결에서는 두 분기의 최종 답변 프롬프트가 같습니다. '
                        '답변을 두 번 생성하면 무작위 변동만 비교하게 되어 API 호출을 생략합니다.')
            else:
                with st.expander('프롬프트 차이 확인'):
                    l_prompt, r_prompt = st.columns(2)
                    l_prompt.text_area('원래 선택의 프롬프트',
                                       prepared['original_messages'][0]['content'],
                                       height=250, disabled=True)
                    r_prompt.text_area('대안 선택의 프롬프트',
                                       prepared['alternative_messages'][0]['content'],
                                       height=250, disabled=True)
                try:
                    api_key = str(st.secrets.get('OPENAI_API_KEY', '')).strip()
                except Exception:
                    api_key = ''
                if not api_key:
                    st.info('Streamlit Secrets에 OPENAI_API_KEY를 설정하면 답변 비교를 실행할 수 있습니다.')
                control = st.checkbox('같은 경로 재생성 대조군 포함', value=True)
                st.caption('대조군 포함 시 3회, 제외 시 2회 호출합니다. 문자열 차이는 품질 점수가 아닙니다.')
                model = st.text_input('모델', value='gpt-4o-mini')
                signature = hashlib.sha256(repr((name, tid, ri, bi, alternative,
                                                wiring, model, control,
                                                prepared['original_prompt_hash'],
                                                prepared['alternative_prompt_hash'])).encode()).hexdigest()
                if st.button('두 답변 생성해 비교', disabled=not bool(api_key.strip())):
                    try:
                        from openai import OpenAI
                        with st.spinner('원래 선택과 대안 선택에 각각 같은 조건으로 요청 중…'):
                            probe = intervention_control(OpenAI(api_key=api_key.strip(), max_retries=0, timeout=45), prepared,
                                                  model=model.strip()) if control else generate_pair(OpenAI(api_key=api_key.strip(),max_retries=0,timeout=45), prepared,
                                                  model=model.strip())
                        st.session_state['answer_probe'] = (signature, probe)
                    except Exception as exc:
                        status = getattr(exc, 'status_code', None)
                        st.error(f'모델 호출 실패: {type(exc).__name__}'
                                 + (f' (HTTP {status})' if status else '')
                                 + '. Secrets의 키 권한과 모델 이름을 확인해주세요.')
                cached = st.session_state.get('answer_probe')
                if cached and cached[0] == signature and cached[1].get('status') == 'error':
                    st.error('일부 모델 호출이 실패했습니다. 성공한 비교로 집계하지 않습니다.')
                    st.json(cached[1])
                if cached and cached[0] == signature and cached[1].get('status') == 'generated':
                    probe = cached[1]
                    first, second = st.columns(2)
                    first.markdown('**원래 선택으로 새로 생성한 답변**')
                    first.write(probe['original_answer'])
                    second.markdown('**대안 선택으로 새로 생성한 답변**')
                    second.write(probe['alternative_answer'])
                    st.write('답변 문자 그대로 일치:', not probe['answer_changed'])
                    st.write('문자열 유사도:', probe['text_similarity'])
                    if control:
                        st.write('같은 경로 재생성 대조군:', probe['control_answer'])
                        st.write('원래 답변과 대조군의 문자열 유사도:', probe['control_similarity'])
                        st.json({k:probe[k] for k in ('calls','input_tokens','output_tokens','cached_tokens','api_seconds')})
                        st.caption('단정 여부·판단 결과·근거 요청이 어떻게 바뀌었는지 직접 평가하세요. '
                                   '문자열 유사도 차이만으로 인과성을 판정하지 않습니다.')
                    st.caption('실제 기록된 당시 답변과 위 두 답변은 별개입니다. '
                               '한 번의 답변 차이만으로 정확도 향상이나 인과적 충실도를 확정할 수 없습니다. '
                               '생성 시 temperature=0을 사용하지만 모델 출력은 완전 결정적이지 않을 수 있습니다.')
                    st.download_button('답변 비교 JSON 받기',
                                       __import__('json').dumps({
                                           'snapshot': name, 'tree': tid,
                                           'original_path': result['original_path'],
                                           'alternative_path': result['alternative_path'],
                                           'prompt_changed': prepared['prompt_changed'],
                                           'original_prompt_hash': prepared['original_prompt_hash'],
                                           'alternative_prompt_hash': prepared['alternative_prompt_hash'],
                                           **probe}, ensure_ascii=False, indent=2),
                                       file_name='traceable_answer_intervention.json',
                                       mime='application/json')
            import json
            st.download_button('실험 결과 JSON 받기', json.dumps(result, ensure_ascii=False, indent=2),
                               file_name='traceable_intervention.json', mime='application/json')
        except Exception as exc:
            st.error(f'기록과 저장 시점 그래프를 연결할 수 없습니다: {exc}')

with tab_benchmark:
    render_benchmark(snapshots)

with tab_quality:
    render_quality(snapshots)

with tab_objective:
    render_objective(snapshots)
