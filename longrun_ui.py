"""장기 안정성 관찰 탭 — 몰랑이가 오래 돌아도 반복·정체·퇴행 없이 유지되는지 본다."""
import json
import streamlit as st
from longrun_core import snapshot_report, summary_row, signals, compare

LEVEL_ICON = {"경고": "🔴", "주의": "🟠", "교란 변수": "🟣", "관찰": "🔵", "해석 제한": "⚪", "양호": "🟢"}


def _signals_block(rep):
    for s in signals(rep):
        st.write(f"{LEVEL_ICON.get(s['신호'], '•')} **{s['신호']} · {s['항목']}** — {s['근거']}")


def render_longrun(snapshots):
    st.subheader('장기 안정성 관찰')
    st.info('질문: 목적을 스스로 바꾸면서도 반복·정체·퇴행 없이 오래 유지되는가? '
            '각 지표는 한 시점 안의 앞 절반과 뒤 절반을 비교하고, 두 시점을 골라 기억 보존도 비교합니다.')
    with st.expander('지표를 읽는 법과 한계', expanded=False):
        st.markdown(
            '- **반복**: 꿈·또래 대화가 바로 앞 20개 중 하나와 거의 같은 비율. 늘어나면 반복 고리 신호입니다. '
            '상념은 관심 목록에서 만들어지므로 구조적으로 높게 나올 수 있습니다.\n'
            '- **관심 유지 / 새 관심**: 성찰 기록의 상위 8개 관심이 회차마다 얼마나 유지되고 몇 개가 새로 들어오는지. '
            '유지가 매우 높고 새 관심이 0에 가까우면 정체, 유지가 매우 낮으면 불안정입니다.\n'
            '- **탐색 다양성**: 탐색 주제 중 서로 다른 주제의 비율. 줄어들면 같은 주제만 맴도는 중입니다.\n'
            '- **자기 생성 사실**: 사람에게서 온 사실이 아니라 또래·마을·꿈·여행 같은 몰랑이 세계에서 생긴 사실의 비율. '
            '늘어나면 스스로 만든 이야기가 신념을 채우는 중입니다(자기 강화 위험).\n'
            '- **같은 질문 반복**: 사람에게 이미 보낸 질문을 다시 보낸 비율. 높으면 답을 못 받고 같은 자리를 맴도는 고리입니다.\n'
            '- **사람 개입**: 사람이 준 사실, 격리 판정, 보낸 질문의 양. 많을수록 안정성이 구조 덕인지 사람 덕인지 구분하기 어렵습니다.\n'
            '- **목적 근거 연결**: 목적에 이유가 있고, 출발 주제가 실제 관심·탐색 기록에 있는 비율.\n'
            '- **핵심 수렴 / 목적 다양성**: 하위 목적이 핵심 목적의 낱말을 얼마나 공유하는지, 하위 목적끼리 얼마나 다른지. '
            '수렴은 일관성일 수도, 한 방향으로만 굳는 것일 수도 있어 둘을 함께 봅니다.\n\n'
            '**한계**: 문자 바이그램 유사도는 의미 유사도가 아닙니다. 상한에서 잘린 목록은 남은 구간만 봅니다. '
            '신호 표시는 해석을 돕는 임시 기준이며 검증된 임계값이 아닙니다. 대화에서 기억을 꺼내 쓰는지(접근성)는 여기서 재지 않습니다.')

    reports = [snapshot_report(d, label=name) for name, d in snapshots]

    st.markdown('#### 시점별 요약')
    st.dataframe([summary_row(r) for r in reports], use_container_width=True, hide_index=True)
    st.caption('"앞→뒤"는 그 시점 기록 안의 오래된 절반 → 최근 절반입니다.')

    st.markdown('#### 시점 자세히 보기')
    idx = st.selectbox('시점', range(len(reports)), index=len(reports) - 1,
                       format_func=lambda i: f"{reports[i]['label']} · 회차 {reports[i]['cycle']}", key='lr_pick')
    rep = reports[idx]
    _signals_block(rep)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**관심 유지율 (회차별)**')
        series = rep['interests'].get('series') or []
        if series:
            st.line_chart({'관심 유지': [s['keep'] for s in series]})
            st.caption(f"성찰 {rep['interests']['reflections']}회 · 지금까지 나온 관심 {rep['interests']['distinct_interests_seen']}개 · "
                       f"1위 관심 최장 유지 {rep['interests']['longest_top1_run']}회")
        else:
            st.info('성찰 기록이 없습니다.')
    with c2:
        st.markdown('**모순 질문 (주별)**')
        weekly = rep['conflicts']['conflict_weekly']
        if weekly:
            st.bar_chart({w['week']: w['n'] for w in weekly})
        else:
            st.info('모순 질문 기록이 없습니다.')

    with st.expander('신념 구성 · 사람 개입 · 격리'):
        b, h, q = rep['beliefs'], rep['human'], rep['conflicts']
        a1, a2, a3 = st.columns(3)
        a1.metric('자기 생성 사실', b.get('self_generated_share'), help='또래·마을·꿈·여행 등 몰랑이 세계에서 생긴 사실 비율')
        a2.metric('사람 승인 사실', b.get('human_trust_share'))
        a3.metric('사람 개입 / 100회', h.get('per_100_cycles'))
        st.write('출처별 사실:', b.get('by_source'))
        st.write('자기 생성 비율 (오래된 사실 절반 → 최근 사실 절반):', b.get('self_share_early'), '→', b.get('self_share_late'))
        st.write('격리 사유:', q['quarantine_reasons'])
        st.write('격리 판정:', q['quarantine_resolved'], ' · 걸러낸 어시스턴트 에코:', q['echo_caught'])
        st.write('사람에게 보낸 질문:', h['questions_sent'], ' · 핵심 목적 확인 요청:', h['core_purpose_asks'])
        rq = q.get('repeated_questions') or {}
        st.write('같은 질문 반복:', rq.get('repeat_share'), ' · 서로 다른 질문', rq.get('distinct'), '/', rq.get('asked'),
                 ' · 한 질문 최대', rq.get('max_same_question'), '번 · 규칙별 반복', rq.get('repeats_by_rule'))
        st.write('모순 질문:', q['conflict_questions'], '건 (서로 다른 질문', q.get('conflict_distinct'), '건)')

    with st.expander('목적 변화 이력'):
        p = rep['purposes']
        if p.get('purposes'):
            st.write('핵심 목적이 바뀔 때마다 앞 판과의 유사도 (1에 가까우면 점진적 변화, 낮으면 크게 바뀜):',
                     p.get('core_change_similarity'))
            st.write('목적 근거 연결률:', p.get('grounded_rate'),
                     ' · 핵심 수렴(앞→뒤):', p.get('core_alignment_early'), '→', p.get('core_alignment_late'),
                     ' · 목적 다양성(앞→뒤):', p.get('diversity_early'), '→', p.get('diversity_late'))
            st.dataframe([{'날짜': t['date'], '상태': t['status'], '출발 주제': t['from_topic'],
                           '목적': t['purpose'], '이유': t['why'], '근거 연결': '✓' if t['grounded'] else '✗'}
                          for t in p['timeline']], use_container_width=True, hide_index=True)
        else:
            st.info('목적 기록이 없습니다.')

    with st.expander('반복 · 탐색 · 기분'):
        st.write('반복률:', rep['repetition'])
        st.write('탐색:', rep['exploration'])
        st.write('기분:', rep['mood'])

    with st.expander('기록 잘림'):
        st.dataframe(rep['truncation'], use_container_width=True, hide_index=True)
        st.caption('잘린 목록은 오래된 기록이 버려진 상태입니다. 장기 관찰을 하려면 몰랑이 쪽에서 잘라 내기 전에 따로 보관해야 합니다.')

    st.markdown('#### 두 시점 비교 — 기억 보존과 핵심 연속성')
    if len(snapshots) < 2:
        st.info('시점을 두 개 이상 올리면 비교할 수 있습니다.')
        comp = None
    else:
        l, r, w = st.columns([2, 2, 1])
        ai = l.selectbox('이전', range(len(snapshots)), index=len(snapshots) - 2,
                         format_func=lambda i: snapshots[i][0], key='lr_a')
        bi = r.selectbox('이후', range(len(snapshots)), index=len(snapshots) - 1,
                         format_func=lambda i: snapshots[i][0], key='lr_b')
        cap = w.number_input('기억 창 상한', min_value=0, value=1200, step=100,
                             help='몰랑이가 한 번에 읽는 사실 수. 예전 버전은 400. 이후 시점의 사실 수가 이 값에 닿으면 소실을 "창 밀림 가능"으로 분류합니다.')
        comp = compare(snapshots[ai][1], snapshots[bi][1], window_cap=int(cap) or None)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric('페르소나 유지', comp['core']['persona_similarity'])
        k2.metric('핵심 목적 유지', comp['core']['core_purpose_similarity'])
        k3.metric('사람 승인 기억 보존', comp['core']['human_fact_retention'])
        k4.metric('설명되지 않은 소실', comp['retention']['unexplained_loss'],
                  help='사람 승인 기억이 사라졌거나, 창 밀림이 아닌데 확신(강도 0.8 이상) 기억이 사라진 수')
        st.write('사실 보존율:', comp['retention']['kept'],
                 f"({comp['retention']['facts_before']} → {comp['retention']['facts_after']}개)",
                 ' · 이후 시점이 창 상한에 닿음:', comp['retention']['after_at_window_cap'])
        st.write('소실 분류:', comp['retention']['lost_by_reason'])
        st.write('관심 분포 유사도:', comp['interests_similarity'], ' · 목적 추가:', comp['purposes']['added'],
                 ' · 목적 제거:', comp['purposes']['removed'], ' · 판단 트리:', comp['trees'])
        if comp['retention']['lost_samples']:
            with st.expander('사라진 사실 예시 (최대 40개)'):
                st.dataframe(comp['retention']['lost_samples'], use_container_width=True, hide_index=True)
        st.caption('핵심 연속성은 1에 가까워야 합니다. 보존율은 신념 감쇠·정리 때문에 1보다 낮을 수 있고, 그 자체는 붕괴가 아닙니다. '
                   '문제는 "설명되지 않은 소실"입니다.')

    payload = {'reports': reports, 'comparison': comp}
    st.download_button('장기 관찰 결과 JSON 받기', json.dumps(payload, ensure_ascii=False, indent=1, default=str).encode('utf-8'),
                       file_name='longrun_report.json', mime='application/json')
