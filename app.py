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

st.set_page_config(page_title='Traceable · 몰랑이', page_icon='🌳', layout='wide')
st.title('Traceable · 몰랑이 기록 분석')
st.caption('피클 시점 비교와 로컬 분기 개입 실험')

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

files = st.file_uploader('몰랑이 성장 기록 (.pkl) — 여러 시점 선택 가능', type=['pkl'], accept_multiple_files=True)
snapshots = []
for uploaded in files:
    try:
        snapshots.append((uploaded.name, read_snapshot(uploaded.getvalue())))
    except Exception as exc:
        st.error(f'{uploaded.name}: {exc}')
if not snapshots:
    st.info('몰랑이 성장 기록을 올리면 분석이 시작됩니다.')
    st.stop()
snapshots.sort(key=lambda pair: int(pair[1].get('meta', {}).get('cycle', 0)))
tab_scores, tab_intervene = st.tabs(['시점별 지표', '분기 개입 실험'])
with tab_scores:
    scores = series([from_molang(d, label=name) for name, d in snapshots])
    rows = [{'파일': s['label'], '회차': s['step'], '사실 수': s['size'],
             **{AXIS_KO[k]: s[k] for k in AXIS_KO}} for s in scores]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption('0~1 비율. 빈 값은 미측정입니다. 집계 단위는 판단 건수가 아니라 트리입니다.')
    st.warning('이 점수는 XAI의 인과적 충실도 점수가 아닙니다. 근거 연결률은 트리의 기억 보유, 학습 반영률은 현재 가중치의 편차, 상태 유래율은 단어 겹침을 측정합니다. 출처에는 기본값이 포함될 수 있습니다.')
    st.caption('현재 몰랑이 어댑터는 replayable을 설정하지 않아 재현 가능률이 0으로 나올 수 있습니다. 이는 재현 실패율을 뜻하지 않습니다. 연속성은 확신한 동일 사실 문장의 유지율입니다.')
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
            st.warning('실제 LLM 답변은 생성하지 않습니다. 경로·지시문·로컬 학습의 변화만 비교하며, 최종 발화의 변화는 입증하지 않습니다.')
            import json
            st.download_button('실험 결과 JSON 받기', json.dumps(result, ensure_ascii=False, indent=2),
                               file_name='traceable_intervention.json', mime='application/json')
        except Exception as exc:
            st.error(f'기록과 저장 시점 그래프를 연결할 수 없습니다: {exc}')
