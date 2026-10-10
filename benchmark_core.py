"""Frozen snapshot benchmark. No DB, learning, or model-priced assumptions."""
from __future__ import annotations
import copy
import hashlib
import json
import random
import re
import time
from types import SimpleNamespace
from llm_bridge import make_classifier, make_choose_fn, make_answer_fn, make_consolidator
from thought_structure import ThoughtStructure, JudgmentNode
from groundcheck import check_answer, guard_prompt

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
