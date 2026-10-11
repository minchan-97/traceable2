"""Headless UI contract smoke; no browser or real Streamlit rendering."""
import sys,types,unittest,importlib
from unittest.mock import patch
from test_benchmark import snap
from test_objective import Fake,suite
from objective_core import new_run,execute
class Stub(types.ModuleType):
    def __init__(self,run):
        super().__init__('streamlit');self.session_state={'objective_run':run};self.secrets={};self.downloads=[]
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def __getattr__(self,name):
        def f(*args,**kw):
            if name=='columns':return [self]*args[0]
            if name in ('expander','empty','progress','form'):return self
            if name=='checkbox':return kw.get('key','').startswith('obj_reveal') or kw.get('value',False)
            if name in ('selectbox','radio'):return list(args[1])[kw.get('index',0)]
            if name=='number_input':return kw.get('value',args[3] if len(args)>3 else args[1] if len(args)>1 else 0)
            if name=='multiselect':return kw.get('default',[])
            if name=='text_input':return args[1] if len(args)>1 else kw.get('value','')
            if name in ('button','form_submit_button'):return False
            if name=='file_uploader':return None
            if name=='download_button':self.downloads.append(kw['file_name']);return False
        return f
class Tests(unittest.TestCase):
    def test_objective_result_controls(self):
        r=new_run(snap(),suite(),'structure','fake',1,20,'toy',['r','f','a'],['r','f','b']);execute(Fake(),r,snap())
        st=Stub(r)
        with patch.dict(sys.modules,{'streamlit':st}):
            sys.modules.pop('quality_ui',None);sys.modules.pop('objective_ui',None)
            module=importlib.import_module('objective_ui');module.render_objective([('test',snap())])
        self.assertIn('objective_blind_review.json',st.downloads)
        self.assertIn('traceable_objective_results.json',st.downloads)
        self.assertIn('traceable_objective_summary.csv',st.downloads)
if __name__=='__main__':unittest.main()
