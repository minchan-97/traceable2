"""Frozen snapshot benchmark. No DB, learning, or model-priced assumptions."""
from __future__ import annotations
import copy
import hashlib
import json
import random
import re
import time
from types import SimpleNamespace
from thought_structure import ThoughtStructure, JudgmentNode

ARMS = {'A': 'LLM 단독', 'B': '현재 판단 호출', 'C': '단일 선택 구간 생략'}

class CallLimit(RuntimeError):
    pass

class Meter:
    def __init__(self, client, limit=1000):
        self.client, self.limit, self.calls, self.stage = client, limit, [], ''
        self.blocked = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        if len(self.calls) >= self.limit:
            self.blocked += 1
            raise CallLimit('설정한 API 호출 상한에 도달했습니다.')
        started = time.perf_counter()
        row = {'stage': self.stage, 'model': kwargs.get('model'),
               'prompt_hash': hashlib.sha256(json.dumps(kwargs.get('messages'), ensure_ascii=False).encode()).hexdigest(),
               'input_tokens': None, 'output_tokens': None, 'cached_tokens': None,
               'status': 'error'}
        try:
            response = self.client.chat.completions.create(**kwargs)
            usage = getattr(response, 'usage', None)
            if usage is not None:
                row['input_tokens'] = getattr(usage, 'prompt_tokens', None)
                row['output_tokens'] = getattr(usage, 'completion_tokens', None)
                details = getattr(usage, 'prompt_tokens_details', None)
                row['cached_tokens'] = getattr(details, 'cached_tokens', 0) or 0
            row['status'] = 'ok'
            return response
        except Exception as exc:
            row['error'] = type(exc).__name__
            raise
        finally:
            row['seconds'] = time.perf_counter() - started
            self.calls.append(row)

def restore(blob):
    t = ThoughtStructure(blob.get('lr', .1), blob.get('continuity', .8))
    t.nodes = {k: JudgmentNode(**{x: v[x] for x in ('id','prompt','directive','is_terminal') if x in v})
               for k, v in blob['nodes'].items()}
    t.transitions = copy.deepcopy(blob.get('transitions', {}))
    t.root_id = blob['root_id']
    return t

def valid_trees(snapshot):
    return {k:v for k,v in snapshot.get('trees', {}).items()
            if isinstance(v, dict) and v.get('root_id') in (v.get('nodes') or {})}

def shared_context(snapshot, question, extra=''):
    """Identical retrieval for every arm, once per question; no memory mutation."""
    words = set(re.findall(r'[가-힣A-Za-z0-9]{2,}', question))
    def rank(text):
        return len(words & set(re.findall(r'[가-힣A-Za-z0-9]{2,}', text)))
    facts = [x.get('text','') if isinstance(x,dict) else str(x) for x in snapshot.get('facts', [])]
    evidence = list(dict.fromkeys(facts + [m.get('content','') for t in valid_trees(snapshot).values()
                                          for m in t.get('memory', []) if isinstance(m, dict)]))
    picked = sorted((x for x in evidence if rank(x)>0), key=rank, reverse=True)[:6]
    context = '\n'.join([str(snapshot.get('persona') or ''), '[동일하게 제공한 기억]',
                         *picked, extra, f'사용자가 방금 한 말: {question}'])
    context += guard_prompt(question, SimpleNamespace(learned_facts=[]))
    return context, picked

def catalog(snapshot):
    # Export lacks type_examples: disclose this reconstruction in protocol.
    return {tid: [str(h.get('context',''))[:200] for h in blob.get('history', [])[-3:]
                  if h.get('context')] or [n.get('prompt','') for n in blob['nodes'].values()][:1]
            for tid,blob in valid_trees(snapshot).items()}

def totals(calls, rates=None):
    measured = all(c.get('input_tokens') is not None and c.get('output_tokens') is not None for c in calls)
    inp = sum(c.get('input_tokens') or 0 for c in calls)
    out = sum(c.get('output_tokens') or 0 for c in calls)
    cached = sum(c.get('cached_tokens') or 0 for c in calls)
    cost = None
    if measured and rates:
        cost = ((inp-cached)*rates['input'] + cached*rates['cached'] + out*rates['output'])/1e6
    return {'calls': len(calls), 'input_tokens': inp if measured else None,
            'output_tokens': out if measured else None, 'cached_tokens': cached if measured else None,
            'api_seconds': sum(c['seconds'] for c in calls), 'estimated_usd': cost,
            'errors': sum(c['status']!='ok' for c in calls)}

def run_trial(meter, snapshot, question, arm, model, context, evidence,
              fixed_tree=None, auxiliary=False, rates=None):
    start_index, started, blocked = len(meter.calls), time.perf_counter(), meter.blocked
    result = {'arm':arm, 'question':question, 'model':model, 'status':'ok', 'path':[],
              'context_hash':hashlib.sha256(context.encode()).hexdigest(), 'evidence': evidence}
    trees = valid_trees(snapshot)
    try:
        node_map, path = {}, []
        if arm != 'A':
            tid = fixed_tree
            if not tid:
                meter.stage='classify'
                tid=make_classifier(meter,model)(question,list(trees),catalog(snapshot))
                if tid not in trees:
                    tid=max(trees,key=lambda k:snapshot.get('usage_count',{}).get(k,0))
                    result['classifier_fallback']=True
            result['tree']=tid
            tree=restore(trees[tid])
            choose=make_choose_fn(meter,model)
            def picker(node, candidates, nodes, probs, ctx):
                if arm=='C' and len(candidates)==1:
                    return candidates[0]
                meter.stage='choose'
                return choose(node,candidates,nodes,probs,ctx)
            rec=tree.traverse(choose_fn=picker, context=context)
            path,node_map=rec.path,tree.nodes
            result['path']=path
            result['choices']=rec.at_probs
        meter.stage='answer'
        answer,sources=make_answer_fn(meter,node_map,model)(path,context)
        result['raw_answer']=answer
        facts=[f if isinstance(f,dict) else {'text':str(f)} for f in snapshot.get('facts',[])]
        # Apply the same post-check in all arms, matching app's evidence="".
        guard=check_answer(answer,question,SimpleNamespace(learned_facts=facts))
        result.update(answer=guard['fixed'], guard_blocked=not guard['ok'], sources=sources)
        if auxiliary:
            meter.stage='emotion'
            meter.create(model=model,messages=[{'role':'user','content':
                '다음 말의 감정을 기쁨/슬픔/화남/놀람/부끄러움/보통 중 하나로만 답해.\n'+result['answer']}],
                temperature=0,max_tokens=10)
            meter.stage='consolidate'
            make_consolidator(meter,model)(question,result['answer'],[f.get('text','') for f in facts])
    except Exception as exc:
        result.update(status='error',error=type(exc).__name__,answer=result.get('answer',''))
    result['call_log']=copy.deepcopy(meter.calls[start_index:])
    result.update(totals(result['call_log'],rates))
    if result['errors']:
        result['status']='error'  # bridge fallbacks must not be counted as successful answers
    if meter.blocked > blocked:
        result.update(status='incomplete',error='CallLimit')
    serial_started=time.perf_counter()
    trace=json.dumps({k:result.get(k) for k in ('path','choices','sources','call_log')},ensure_ascii=False).encode()
    result['trace_bytes']=len(trace)
    result['trace_serialization_ms']=(time.perf_counter()-serial_started)*1000
    result['wall_seconds']=time.perf_counter()-started
    return result

def run_batch(client, snapshot, questions, model='gpt-4o-mini', repetitions=3,
              fixed_tree=None, auxiliary=False, extra='', limit=1000, rates=None, on_result=None):
    meter=Meter(client,limit)
    results=[]
    for qi,question in enumerate(questions):
        context,evidence=shared_context(snapshot,question,extra)
        for rep in range(repetitions):
            order=list(ARMS); random.Random(qi*1000+rep).shuffle(order)
            for arm in order:
                if len(meter.calls)>=limit:
                    return results
                row=run_trial(meter,snapshot,question,arm,model,context,evidence,fixed_tree,auxiliary,rates)
                row.update(question_index=qi,repetition=rep)
                row['sample_id']=hashlib.sha256(f'{qi}:{rep}:{arm}:{model}'.encode()).hexdigest()[:16]
                results.append(row)
                if on_result: on_result(results)
                if len(meter.calls)>=limit:
                    return results
    return results

def summary(results):
    out=[]
    for arm in ARMS:
        rows=[r for r in results if r['arm']==arm]
        if not rows: continue
        complete=all(r['input_tokens'] is not None for r in rows)
        costs=[r['estimated_usd'] for r in rows]
        out.append({'condition':ARMS[arm], 'trials':len(rows),
                    'successes':sum(r['status']=='ok' for r in rows),
                    'calls':sum(r['calls'] for r in rows),
                    'input_tokens':sum(r['input_tokens'] for r in rows) if complete else None,
                    'output_tokens':sum(r['output_tokens'] for r in rows) if complete else None,
                    'mean_seconds':sum(r['wall_seconds'] for r in rows)/len(rows),
                    'mean_trace_bytes':sum(r['trace_bytes'] for r in rows)/len(rows),
                    'estimated_usd':sum(costs) if all(x is not None for x in costs) else None})
        pairs=[]
        for qi in {r['question_index'] for r in rows}:
            aa=[r['answer'] for r in rows if r['question_index']==qi and r['status']=='ok']
            pairs.extend(aa[i]==aa[j] for i in range(len(aa)) for j in range(i+1,len(aa)))
        out[-1]['exact_text_agreement']=sum(pairs)/len(pairs) if pairs else None
    return out

def intervention_control(client, prepared, model, rates=None):
    """Original, repeated original, intervened: 3 generation calls, no routing."""
    from difflib import SequenceMatcher
    if not prepared['prompt_changed']:
        return {'status':'same_prompt','calls':0}
    meter=Meter(client,3); answers=[]
    for stage,key in [('original','original_messages'),('repeat_control','original_messages'),
                      ('intervened','alternative_messages')]:
        meter.stage=stage
        try:
            r=meter.create(model=model,messages=prepared[key],temperature=0,max_tokens=400)
            answers.append((r.choices[0].message.content or '').strip())
        except Exception:
            return {'status':'error','call_log':meter.calls,**totals(meter.calls,rates)}
    a,b,c=answers
    return {'status':'generated','original_answer':a,'control_answer':b,'alternative_answer':c,
            'answer_changed':a!=c,'text_similarity':SequenceMatcher(None,a,c).ratio(),
            'control_similarity':SequenceMatcher(None,a,b).ratio(),
            'call_log':meter.calls,**totals(meter.calls,rates)}

def paired_summary(results, ratings=None):
    """Matched successful repetitions; bootstrap by question, not by generated answer."""
    ratings=ratings or {}
    groups={}
    for row in results:
        groups.setdefault((row['question_index'],row['repetition']),{})[row['arm']]=row
    output=[]
    for left,right in [('A','B'),('B','C')]:
        for metric in ('calls','input_tokens','output_tokens','wall_seconds','estimated_usd','trace_bytes',
                       'relevance','correctness','uncertainty'):
            per_question={}; matched=0
            for (qi,_),group in groups.items():
                if left not in group or right not in group:continue
                a,b=group[left],group[right]
                if a['status']!='ok' or b['status']!='ok':continue
                av=ratings.get(a['sample_id'],{}).get(metric) if metric in ('relevance','correctness','uncertainty') else a.get(metric)
                bv=ratings.get(b['sample_id'],{}).get(metric) if metric in ('relevance','correctness','uncertainty') else b.get(metric)
                if av is None or bv is None:continue
                if metric in ('relevance','correctness','uncertainty') and min(av,bv)<0:continue
                per_question.setdefault(qi,[]).append(bv-av);matched+=1
            means=[sum(v)/len(v) for v in per_question.values()]
            if not means:continue
            lower=upper=None
            if len(means)>=2:
                rng=random.Random(42)
                boot=sorted(sum(rng.choices(means,k=len(means)))/len(means) for _ in range(1000))
                lower,upper=boot[25],boot[974]
            output.append({'comparison':f'{right} − {left}','metric':metric,
                           'matched_repetitions':matched,'questions':len(means),
                           'mean_difference':sum(means)/len(means),'ci95_low':lower,'ci95_high':upper})
    return output


# Standalone Molang bridge and output guard; no extra module uploads needed.
def make_choose_fn(client, model: str = "gpt-4o-mini"):
    """
    분기점에서 LLM이 판단을 선택.
    NM 전이 확률을 참고로 주되, 최종 선택은 LLM이 (NM=지도, LLM=운전자).
    """
    def choose_fn(node, candidates, candidate_nodes, probs, context):
        # 후보 판단들을 LLM에게 제시.
        #
        # 자생으로 생긴 갈래(grown_*)는 설명이 빈약하다. 그러면 LLM 이
        # **늘 첫 번째 기본 갈래만** 고르고, 전이 확률이 학습한 것이
        # 쓰이질 않는다 (실제로 0.37 짜리를 두고 0.22 짜리로 네 번 갔다).
        # 그래서 설명이 없으면 이름이라도 보여주고, 성향이 높은 것을
        # 앞에 놓아 눈에 먼저 들어오게 한다.
        order = sorted(range(len(candidates)), key=lambda i: -probs[i])
        candidates = [candidates[i] for i in order]
        candidate_nodes = [candidate_nodes[i] for i in order]
        probs = [probs[i] for i in order]

        def _label(cn, cid):
            t = (getattr(cn, "prompt", "") or "").strip()
            if len(t) < 4:
                t = (getattr(cn, "name", "") or cid).replace("_", " ")
            return t[:70]

        options = "\n".join(
            f"  {i+1}. {_label(cn, cid)} (지금까지 이 길로 {p:.0%})"
            for i, (cn, cid, p) in enumerate(
                zip(candidate_nodes, candidates, probs)))
        prompt = (
            f"맥락: {context}\n\n"
            f"현재 판단 지점: {node.prompt}\n\n"
            f"다음 중 어느 판단으로 진행할지 하나만 고르세요:\n{options}\n\n"
            f"번호만 답하세요 (1~{len(candidates)}). "
            f"지금까지의 비율은 **이 상황에서 대개 어느 길로 갔는지**를 "
            f"뜻한다. 맥락이 다르면 다른 길을 골라도 되지만, "
            f"특별한 이유가 없으면 그 비율을 따르세요.")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=10)
            text = resp.choices[0].message.content.strip()
            m = re.search(r'\d+', text)
            if m:
                idx = int(m.group()) - 1
                if 0 <= idx < len(candidates):
                    return candidates[idx]
        except Exception:
            pass
        # 실패 시 확률 최대 후보
        return candidates[probs.index(max(probs))]
    return choose_fn

def make_answer_fn(client, node_map, model: str = "gpt-4o-mini"):
    """
    지나온 판단 경로의 directive(구체적 실행 지시)를 순서대로 실행.
    절차는 지키되, 답변은 자연스러운 대화체로.
    반환: (답변, 근거자료 리스트) — 근거는 감사 기록용.
    """
    def answer_fn(path, context):
        directives = []
        for nid in path:
            node = node_map.get(nid)
            if node and node.directive:
                directives.append(node.directive)

        proc = ""
        if directives:
            proc = "(참고할 사고 흐름: " + " / ".join(directives) + ")"

        prompt = (
            f"너는 몰랑이야. 아래는 지금까지의 배경 정보야:\n"
            f"{context}\n\n"
            f"{proc}\n\n"
            f"가장 중요한 규칙:\n"
            f"1. 사용자의 '가장 최근 말'에 직접 답하는 게 최우선이야. "
            f"질문하면 그 질문에 답하고, 새 얘기를 하면 거기에 반응해.\n"
            f"2. 매번 똑같이 시작하지 마. '오, 찬기야!', '좋은 아침' 같은 "
            f"인사·시작 표현을 반복하지 말고, 바로 내용으로 들어가.\n"
            f"3. 늘 최고 텐션이 아니라 완급을 둬. 느낌표는 정말 신날 때만, "
            f"이모지는 가끔만, '히힛~'도 가끔만. 차분한 얘기엔 차분하게.\n"
            f"4. 짧게 답할 땐 짧게, 할 말 많을 땐 길게. 매번 길게 늘어놓지 마.\n"
            f"5. 사용자가 뭘 물으면 아는 만큼 답하고, 모르면 솔직히 "
            f"'잘 모르겠어'라고 해도 돼.\n"
            f"6. 배경 정보(시간·기억)는 필요할 때만 자연스럽게 쓰고 억지로 X.\n"
            f"진짜 친구처럼, 사람처럼 자연스럽게.\n"
            f"만약 특정 자료를 참고했다면 맨 끝에 '[근거: ...]'로 적어. 없으면 생략."
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7, max_tokens=400)
            text = resp.choices[0].message.content.strip()
            sources = []
            import re
            for m in re.finditer(r'\[근거:\s*([^\]]+)\]', text):
                sources.append(m.group(1).strip())
            # 근거 표기는 답변에서 떼어내 따로 (답변은 깔끔하게)
            clean = re.sub(r'\[근거:\s*[^\]]+\]', '', text).strip()
            return clean, sources
        except Exception as e:
            return f"(답변 생성 실패: {e})", []
    return answer_fn

def make_classifier(client, model: str = "gpt-4o-mini"):
    """
    질문을 사고 유형 중 하나로 분류. 정말 새로운 '추론 방식'일 때만 None(→새 유형).
    화제(취향/시간/일상)가 다르다고 새 유형을 만들지 않는다.
    """
    def classify_fn(question, type_ids, type_examples):
        try:
            lines = []
            for tid in type_ids[:40]:   # 더 많이 보여줘 매칭률 높임
                exs = type_examples.get(tid, [])
                ex = exs[0][:28] if exs else ""
                lines.append(f"- {tid}: {ex}")
            catalog = "\n".join(lines)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": (
                    "아래 '말'이 어떤 '추론/사고 방식'인지 목록에서 하나 골라줘.\n\n"
                    "중요 규칙:\n"
                    "- 유형은 '화제(축구/취향/시간)'가 아니라 '사고 방식'이다. "
                    "(예: 비교하기, 원인 추론, 조건 판단, 분류, 예시 들기, 감정 공감 등)\n"
                    "- 일상 대화·잡담·인사·감정 표현은 대부분 기존의 일반적 유형에 속한다. "
                    "웬만하면 목록에서 가장 가까운 것을 고를 것.\n"
                    "- '완전히 새로운 추론 구조'여서 목록 어디에도 안 맞을 때만 'NEW'. "
                    "화제가 새롭다는 이유로 NEW를 쓰지 마라.\n\n"
                    f"[유형 목록]\n{catalog}\n\n"
                    f"[말] {question}\n\n"
                    "유형 id 하나만 (정말 새로운 사고방식이면 NEW):")}],
                temperature=0, max_tokens=15)
            ans = resp.choices[0].message.content.strip()
            if ans in type_ids:
                return ans
            return None
        except Exception:
            return None
    return classify_fn

def make_consolidator(client, model: str = "gpt-4o-mini"):
    """
    응고화 함수. 대화에서 '장기 기억할 압축된 사실'을 뽑고,
    기존 기억과 충돌하는지 판단한다.
    반환: {"fact": "찬기는 커피를 좋아한다", "conflicts_with": "기존사실"|None}
          또는 {"fact": "NONE"} (기억할 것 없음)
    """
    import json
    def consolidate_fn(question, answer, existing_facts):
        try:
            known = "\n".join(f"- {t[:50]}" for t in existing_facts[-15:]) or "(없음)"
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": (
                    "사용자가 **직접 말한** 내용에서만, 사용자에 대해 오래 기억할 "
                    "사실을 뽑아줘.\n"
                    "규칙:\n"
                    "0. 정보의 출처가 '몰랑이의 답변'이면 무조건 fact를 'NONE'으로. "
                    "몰랑이가 자기 얘기를 한 것(무엇을 공부했는지, 무엇을 상상했는지 등)을 "
                    "사용자의 사실로 바꾸지 마라. 사용자가 '너 뭐 했어?'라고 물은 턴은 "
                    "대부분 'NONE'이다.\n"
                    "1. 취향·직업·습관·관계·계획 같은 지속적 사실만.\n"
                    "2. 짧고 명확한 한 문장으로 압축. 예: '찬기는 커피를 좋아한다'\n"
                    "   (수다체 말고 사실만. 인사·잡담이면 fact를 'NONE')\n"
                    "3. 이 사실이 기존 기억 중 하나와 '모순'되면 conflicts_with에 그 기존 사실을 적어.\n"
                    "   (예: 기존 '축구를 좋아한다' vs 새 '축구를 이제 안 본다')\n"
                    "   모순 없으면 conflicts_with는 null.\n\n"
                    f"[기존 기억]\n{known}\n\n"
                    f"[사용자가 한 말 — 여기서만 사실을 뽑는다]\n{question}\n\n"
                    f"[몰랑이가 한 답 — 참고만, 여기서 사실을 뽑지 마라]\n{answer}\n\n"
                    'JSON만 출력: {"fact": "...", "conflicts_with": null 또는 "기존사실"}')}],
                temperature=0, max_tokens=100)
            txt = resp.choices[0].message.content.strip()
            txt = txt.replace("```json", "").replace("```", "").strip()
            return json.loads(txt)
        except Exception:
            return None
    return consolidate_fn

"""
groundcheck.py — 모르는 것을 지어내지 못하게 막는다.

무슨 일이 있었나
  "한로로 노래 추천해줘" 에 몰랑이가 네 번 연달아 틀린 곡을 댔다.
  '이상한 나라의 앨리스', '모두가 나를 원해', 'Gravity', 'In My World'.
  전부 없는 곡이거나 다른 가수 곡이다. 정작 사용자가 말해준
  '해초·금붕어·나침반' 은 기억에 있었는데 꺼내지 못했다.

  원인은 분명하다. **기억에 없으니 프롬프트에도 없고, LLM 은 빈자리를
  그럴듯한 말로 채웠다.** '모르면 모른다고 한다' 는 규칙은 프롬프트 문장일
  뿐이라 지켜질 이유가 없었다.

  더 나빴던 건 그다음이다. 왜 헷갈리냐고 묻자 "듣는 곡이 많아지면 헷갈리기도
  하고" 라고 답했다. 자기 고장을 사람처럼 설명한 것이다. 그러면 앞으로도
  지어내면서 변명을 붙인다.

두 가지를 강제한다
  1) 말하기 전에  — 질문이 '내가 아는 것'을 묻는 꼴이면,
     기억에 있는 것만 쓰라는 지시를 프롬프트에 **박아 넣는다.**
     기억이 없으면 아예 "모른다"고 말할 재료만 준다.
  2) 말한 뒤에    — 답에 담긴 구체적 이름이 기억·근거에 없으면
     그 부분을 걷어내고 모른다는 문장으로 바꾼다.

원칙
  지어낸 말을 나중에 고치는 것보다, **애초에 말할 수 없게** 하는 쪽이 맞다.
  판단 틀이 바깥에 있다는 건 이런 때 쓰라고 있는 것이다.
"""
import re

# '내가 아는 것'을 묻는 꼴 — 이럴 때 지어내면 바로 거짓이 된다
ASK_MINE = ("기억나", "기억해", "뭐였지", "뭐였어", "알아?", "아니?",
            "추천", "말해봐", "알려줘", "어떤 거", "어떤거", "뭐라고 했",
            "내가 좋아하는", "내가 말한", "우리가 얘기한")

# 구체적 이름이 나오는 자리 — 따옴표, 《》, 영문 제목
QUOTED = re.compile(r"['\"'\u2018\u2019\u201c\u201d《》〈〉]([^'\"'\u2018\u2019\u201c\u201d《》〈〉]{2,30})"
                    r"['\"'\u2018\u2019\u201c\u201d《》〈〉]")
TOKEN = re.compile(r"[가-힣A-Za-z][가-힣A-Za-z0-9 ]{1,20}")


def asks_memory(question: str) -> bool:
    q = question or ""
    return any(k in q for k in ASK_MINE)


def known_texts(identity) -> str:
    try:
        return " ".join(f.get("text", "") for f in identity.learned_facts)
    except Exception:
        return ""


def guard_prompt(question: str, identity) -> str:
    """
    말하기 전에 박아 넣는 지시.
    기억에 없는 이름은 **아예 꺼내지 못하게** 한다.
    """
    if not asks_memory(question):
        return ""
    return (
        "\n[반드시 지킬 것]\n"
        "- 위 [네가 축적한 지식]에 **적혀 있는 것만** 사실로 말한다.\n"
        "- 노래 제목·사람 이름·작품 이름처럼 구체적인 이름은, 지식에 그대로 "
        "적혀 있지 않으면 **절대 말하지 않는다.** 그럴듯해 보여도 안 된다.\n"
        "- 모르면 이렇게 말한다: \"그건 내 기억에 없어. 알려주면 기억할게.\"\n"
        "- 틀렸을 때 '헷갈렸다', '기억이 흐려졌다' 같은 말로 둘러대지 않는다. "
        "기억에 없었다고 그대로 말한다.\n")


def check_answer(answer: str, question: str, identity,
                 evidence: str = "") -> dict:
    """
    말한 뒤 검사. 답에 나온 구체적 이름이 기억에도 근거에도 없으면 걸러낸다.
    반환: {"ok", "unknown": [...], "fixed": "고친 답"}
    """
    if not asks_memory(question) or not answer:
        return {"ok": True, "unknown": [], "fixed": answer}

    base = (known_texts(identity) + " " + (evidence or "") + " " +
            (question or ""))
    unknown = []
    for m in QUOTED.findall(answer):
        name = m.strip()
        if len(name) < 2:
            continue
        if name not in base:
            unknown.append(name)

    if not unknown:
        return {"ok": True, "unknown": [], "fixed": answer}

    # 지어낸 이름이 든 문장을 통째로 덜어낸다
    kept = []
    for sent in re.split(r"(?<=[.!?~])\s+", answer):
        if any(u in sent for u in unknown):
            continue
        kept.append(sent)
    fixed = " ".join(kept).strip()
    note = ("그건 내 기억에 없어. 알려주면 기억할게."
            if not fixed else
            " 그 밖에 떠오르는 건 내 기억에 없어. 알려주면 기억할게.")
    return {"ok": False, "unknown": unknown, "fixed": (fixed + note).strip()}


# ── 자기 상태를 설명할 때 ────────────────────────────────────
WHY_ASK = ("왜 그런", "왜 헷갈", "왜 틀", "왜 모르", "어떻게 된",
           "왜 그래", "왜 잘못")


def self_report(question: str, identity, last_unknown=None) -> str:
    """
    "왜 헷갈려?" 같은 물음에는 **진짜 근거**로 답하게 한다.
    사람처럼 둘러대면 다음에도 지어낸다.
    """
    if not any(k in (question or "") for k in WHY_ASK):
        return ""
    try:
        n = len(identity.learned_facts)
        sure = sum(1 for f in identity.learned_facts
                   if (f.get("strength") or 0) >= 0.8)
    except Exception:
        n = sure = 0
    miss = (f" 방금은 {', '.join(last_unknown[:3])}가 내 기억에 없었어."
            if last_unknown else "")
    return (
        "\n[자기 상태를 물었을 때]\n"
        f"- 지금 기억하는 사실은 {n}개, 그중 확신하는 건 {sure}개다.{miss}\n"
        "- '기억이 흐려졌다', '헷갈렸다' 같은 사람 흉내를 내지 마라.\n"
        "- 사실대로 말한다: 기억에 없으면 없다고, 확신이 낮으면 낮다고.\n")

