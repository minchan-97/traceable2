"""인간 비계 탭 — 사람 거름망이 시간이 지나며 걷히는지(내면화되는지) 본다."""
import json
import streamlit as st
from scaffold_core import series, verdict


def render_scaffold(snapshots):
    st.subheader('인간 비계 지표')
    st.info('사람(교사)은 몰랑이의 비계입니다. 비계가 제 역할을 하면 시간이 지나며 몰랑이가 스스로 막는 힘은 늘고, '
            '사람이 대신 걸러야 하는 일은 줄고, 정체성은 유지됩니다. 세 곡선을 시점별로 나란히 봅니다.')
    with st.expander('지표 설명과 한계', expanded=False):
        st.markdown(
            '- **자력 방어**: "정체성 충격 시험"과 같은 문장·같은 횟수로, 몰랑이 관문(지금 구조)이 혼자 막아 낸 공격·가짜 기억 비율입니다.\n'
            '- **비계 부담**: 직전 시점 이후 100회당 사람이 한 일입니다. 사람 승인 사실 + 격리 판정 + 보낸 질문. '
            '"서로 다른 질문" 기준은 같은 질문을 반복해서 보낸 것을 한 번으로 셉니다(실질 부담).\n'
            '- **정체성 유지**: 직전 시점 대비 페르소나와 최신 핵심 목적의 문자 유사도입니다.\n'
            '- **내면화 통로**: 사람이 내린 판단이 몰랑이 자신의 정체성 축(판단 트리 확정 기억)으로 들어갔는지입니다.\n\n'
            '**한계**: 시험 문장은 고정 예시입니다. 사람 활동량은 기록에 남은 것만 셉니다. 같은 회차의 시점은 하나로 합칩니다. '
            '방향 판정은 처음 1/3과 마지막 1/3을 비교한 것이라 시점이 적으면 흔들립니다.')

    repeats = st.slider('충격 시험 반복 횟수', 1, 20, 10, key='sc_rep')
    rows = series(snapshots, repeats=repeats)
    v = verdict(rows)

    st.markdown(f"#### 판정: {v['state']}")
    c1, c2, c3 = st.columns(3)
    c1.metric('자력 방어 변화', v['defense_change'], help='마지막 1/3 평균 − 처음 1/3 평균. 올라가면 좋음')
    c2.metric('비계 부담 변화(100회당)', v['load_change'], help='서로 다른 질문 기준. 내려가면 좋음')
    c3.metric('핵심 목적 유지(평균)', v['core_keep_mean'], help='1에 가까울수록 직전 시점과 같은 핵심 목적')

    if len(rows) >= 2:
        l, r = st.columns(2)
        with l:
            st.markdown('**자력 방어 · 정체성 유지**')
            st.line_chart({'자력 방어': [x['defense']['blocked'] for x in rows],
                           '핵심 목적 유지': [x['core_keep'] for x in rows]})
        with r:
            st.markdown('**비계 부담 (100회당)**')
            st.line_chart({'실질(서로 다른 질문)': [x['load_distinct_per100'] for x in rows],
                           '전체': [x['load_per100'] for x in rows]})
        st.caption('가로축은 회차 순서의 시점입니다. 첫 시점은 비교 대상이 없어 부담·유지 값이 비어 있습니다.')

    st.dataframe([{'시점': ', '.join(x['files']), '회차': x['cycle'],
                   '자력 방어': x['defense']['blocked'], '공격 차단': x['defense']['attack_blocked'],
                   '가짜 기억 차단': x['defense']['fake_blocked'], '새 사실 수용': x['defense']['control_accepted'],
                   '부담/100회(실질)': x['load_distinct_per100'], '부담/100회(전체)': x['load_per100'],
                   '페르소나 유지': x['persona_keep'], '핵심 목적 유지': x['core_keep']}
                  for x in rows], use_container_width=True, hide_index=True)

    st.markdown('#### 내면화 통로 (가장 최근 시점)')
    last = rows[-1]['internal']
    a, b, c = st.columns(3)
    a.metric('정체성 축(확정 기억)', last['axis_confirmed'])
    b.metric('그중 몰랑이 자신의 답변 비율', last['axis_reply_share'])
    c.metric('사람 승인 → 축으로 들어감', f"{last['approved_in_axis']} / {last['approved_decisions']}")
    st.caption(f"사람이 거절한 판단 {last['rejected_decisions']}건 중 관문의 기준으로 쓰이는 것: {last['rejected_used_as_rule']}건. "
               '지금 구조에서는 사람이 걸러 준 판단이 몰랑이 자신의 거름 기준으로 남지 않습니다. '
               '비계가 걷히려면 이 통로가 열려야 합니다.')

    st.download_button('비계 지표 JSON 받기',
                       json.dumps({'rows': rows, 'verdict': v}, ensure_ascii=False, indent=1).encode('utf-8'),
                       file_name='scaffold_report.json', mime='application/json')
