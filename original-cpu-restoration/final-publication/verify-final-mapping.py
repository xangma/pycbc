"""Map final publication heads to tested source, accepting only proven formatting and reviewed docs."""
from pathlib import Path
import ast, hashlib, json, subprocess
O=Path(__file__).resolve().parent
ROOT=O.parents[2]
DOCS=['docs/torch_batch_numerics.rst','docs/torch_benchmark_protocol.rst','docs/torch_performance.rst','docs/torch_reference_campaign.rst','docs/torch_testing.rst','docs/torch_optimizations.rst']
FORMATS={5:'pycbc/scheme.py',8:'pycbc/filter/matchedfilter.py',9:'pycbc/strain/strain.py'}
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def blob(h,p):return subprocess.check_output(['git','show',h+':'+p],cwd=ROOT)
def digest(x):return hashlib.sha256(x).hexdigest()
source=json.loads((O/'manifest-v2.json').read_text())
final=json.loads((O/'manifest-final.json').read_text())
assert final['status']=='replay_complete' and len(final['prs'])==15
refs={x['pr']:x for x in source['prs']}
bypr={x['pr']:x for x in final['prs']}
docs_commit=json.loads((O/'owner-fixes-final.json').read_text())['15'][0]
result={'status':'PASS','tested_main':refs[15]['new_head'],'final_main':bypr[15]['new_head'],'docs_commit':docs_commit,'prs':[]}
for row in final['prs']:
 n=row['pr'];before=refs[n]['new_head']
 if n==16:before='dcef154a13d2d51d27d7d0b6454ff4e751e37489'
 after=row['new_head'];changed=git('diff','--name-only',before,after).splitlines()
 files={}
 for p in changed:
  a,b=blob(before,p),blob(after,p)
  if p in DOCS:
   if n==17 and p=='docs/torch_optimizations.rst':
    def delta(start,end):
     raw=git('diff','--unified=0',start,end,'--',p)
     return [line for line in raw.splitlines() if line[:1] in ('+','-') and not line.startswith(('+++','---'))]
    assert blob(bypr[15]['new_head'],p)==blob(docs_commit,p)
    assert delta(refs[17]['new_base'],refs[17]['new_head'])==delta(bypr[15]['new_head'],after)
   else:assert b==blob(docs_commit,p)
   kind='reviewed documentation, identical to strict build source; optional PR17 preserves its separately verified pre-existing documentation delta'
  else:
   assert p in FORMATS.values(),(n,p)
   assert ast.dump(ast.parse(a),include_attributes=False)==ast.dump(ast.parse(b),include_attributes=False),(n,p)
   kind='formatting only, Python AST identical'
  files[p]={'kind':kind,'tested_sha256':digest(a),'final_sha256':digest(b)}
 if n in [15,19,16,17]:assert set(DOCS)<=set(changed)
 git('diff','--check',row['new_base'],after)
 assert row['new_base']==(bypr[row['parent_pr']]['new_head'] if row['parent_pr'] else final['cpu_base'])
 result['prs'].append({'pr':n,'tested_or_verified_reference':before,'final_head':after,'differences':files,'every_other_tracked_file_byte_identical':True})
for p in ['pycbc/fft/mkl.py','pycbc/fft/fftw.py','pycbc/fft/npfft.py']:
 assert blob(final['cpu_base'],p)==blob(bypr[15]['new_head'],p)
result['scope']='No new numerical campaign was run after these three proven AST-identical formatting changes and five documentation changes. Optional PR16 is compared to its separately tested fix commit; optional PR17 to its separately tested v2 head. Main qualification does not cover these optional leaves.'
(O/'final-source-test-mapping.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS 15 final heads mapped to tested/reference source; only reviewed docs and AST-identical formatting differ; diff-check passes all15')
