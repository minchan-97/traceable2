import copy
import unittest
from types import SimpleNamespace as N
from benchmark_core import run_batch, summary, intervention_control, paired_summary
from answer_probe import prepare_pair
from thought_structure import JudgmentNode

class Fake:
    def __init__(self,usage=True,fail=False):
        self.chat=N(completions=self);self.usage=usage;self.fail=fail;self.count=0
    def create(self,**kw):
        self.count+=1
        if self.fail:raise RuntimeError('test')
        p=kw['messages'][0]['content']
        text='toy' if '유형 id 하나만' in p else '1' if '번호만 답하세요' in p else '답변'
        return N(choices=[N(message=N(content=text))],usage=N(prompt_tokens=100,completion_tokens=10,
                   prompt_tokens_details=N(cached_tokens=20)) if self.usage else None)

def snap():
    def n(k,terminal=False):return {'id':k,'prompt':k,'directive':k+' 지시','is_terminal':terminal}
    return {'persona':'몰랑이','facts':[{'text':'파도는 바다에 있다'}], 'trees':{'toy':{
        'root_id':'r','nodes':{'r':n('r'),'f':n('f'),'a':n('a',True),'b':n('b',True)},
        'transitions':{'r':{'f':1.0},'f':{'a':.7,'b':.3},'a':{},'b':{}},'memory':[]}}}

class Tests(unittest.TestCase):
    def test_usage_and_freeze(self):
        s=snap();before=copy.deepcopy(s);client=Fake()
        rows=run_batch(client,s,['바다'],repetitions=1,fixed_tree='toy',rates={'input':1,'cached':.5,'output':2})
        by={r['arm']:r for r in rows}
        self.assertEqual([by[k]['calls'] for k in 'ABC'],[1,3,2])
        self.assertEqual(len({r['context_hash'] for r in rows}),1)
        self.assertEqual(by['B']['path'],by['C']['path'])
        self.assertAlmostEqual(by['A']['estimated_usd'],.00011)
        self.assertEqual(s,before)
        self.assertEqual(client.count,6)
    def test_classify(self):
        rows=run_batch(Fake(),snap(),['질문'],repetitions=1)
        self.assertEqual({r['arm']:r['calls'] for r in rows},{'A':1,'B':4,'C':3})
    def test_missing_usage_and_failure(self):
        rows=run_batch(Fake(usage=False),snap(),['질문'],repetitions=1)
        self.assertTrue(all(r['input_tokens'] is None and r['estimated_usd'] is None for r in rows))
        rows=run_batch(Fake(fail=True),snap(),['질문'],repetitions=1)
        self.assertTrue(all(r['status']=='error' for r in rows))
    def test_limit(self):
        c=Fake();rows=run_batch(c,snap(),['질문'],repetitions=3,limit=2)
        self.assertLessEqual(c.count,2)
        self.assertEqual(rows[-1]['status'],'incomplete')
    def test_control(self):
        nodes={'a':JudgmentNode('a','a','결론'),'b':JudgmentNode('b','b','유보')}
        p=prepare_pair(['a'],['b'],'질문',nodes); c=Fake()
        r=intervention_control(c,p,'fake')
        self.assertEqual(c.count,3);self.assertIn('control_answer',r)
        p=prepare_pair(['a'],['b'],'질문',{});c=Fake()
        self.assertEqual(intervention_control(c,p,'fake')['calls'],0)
    def test_pairs(self):
        rows=run_batch(Fake(),snap(),['첫 질문','둘 질문'],repetitions=2,fixed_tree='toy')
        result=paired_summary(rows)
        d=next(x for x in result if x['comparison']=='C − B' and x['metric']=='calls')
        self.assertEqual(d['questions'],2)
        self.assertEqual(d['matched_repetitions'],4)
        self.assertEqual(d['mean_difference'],-1)
        self.assertEqual(d['ci95_low'],-1)

if __name__=='__main__':unittest.main()
