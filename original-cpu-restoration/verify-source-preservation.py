"""Verify ancestry, unchanged native source, and original CPU FFT source."""
from pathlib import Path
import ast, subprocess,json,hashlib
O=Path(__file__).resolve().parent
m=json.loads((O/'manifest-v2.json').read_text());base=m['cpu_base'];main=next(x['new_head'] for x in m['prs'] if x['pr']==15)
def git(*a):return subprocess.check_output(['git',*a],text=True).strip()
def blob(h,p):return subprocess.check_output(['git','show',h+':'+p])
def normalized_setup(h):
 t=ast.parse(blob(h,'setup.py'))
 for n in ast.walk(t):
  if hasattr(n,'body') and isinstance(n.body,list):n.body=[x for x in n.body if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and isinstance(x.value.value,str))]
  if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='extras_require' for x in n.targets):
   keep=[(k,v) for k,v in zip(n.value.keys,n.value.values) if not (isinstance(k,ast.Constant) and k.value=='torch')]
   n.value.keys=[k for k,v in keep];n.value.values=[v for k,v in keep]
 return ast.dump(t,include_attributes=False)
r={'original_cpu':base,'tested_main':main,'prs':[],'byte_exact_original_files':{},'native_unchanged':{},'setup_ast_original_except_optional_torch':{}}
for p in ['pycbc/fft/mkl.py','pycbc/fft/fftw.py','pycbc/fft/npfft.py']:
 a,b=blob(base,p),blob(main,p);assert a==b;r['byte_exact_original_files'][p]=hashlib.sha256(a).hexdigest()
for x in m['prs']:
 subprocess.run(['git','merge-base','--is-ancestor',base,x['new_head']],check=True)
 assert subprocess.run(['git','merge-base','--is-ancestor','66789ac4a7468094b0cc3ca1498a1de67e0311f6',x['new_head']]).returncode==1
 subprocess.run(['git','merge-base','--is-ancestor',x['new_base'],x['new_head']],check=True)
 native=git('diff','--name-only',base,x['new_head'],'--','*.pyx','*.pxd','*.pxi','*.c','*.cc','*.cpp','*.h','*.hpp','*.cu','*.cuh','pycbc/lib');assert not native or x['pr']==17,(x['pr'],native)
 assert not git('diff','--name-only',x['old_head'],x['new_head'],'--','*.pyx','*.pxd','*.pxi','*.c','*.cc','*.cpp','*.h','*.hpp','*.cu','*.cuh','pycbc/lib')
 assert normalized_setup(base)==normalized_setup(x['new_head']),x['pr']
 r['prs'].append({k:x[k] for k in ('pr','new_base','new_head','parent_pr')});r['native_unchanged'][str(x['pr'])]={'restoration':True,'versus_original':not bool(native),'preexisting_optional_native_files':native.splitlines()};r['setup_ast_original_except_optional_torch'][str(x['pr'])]=True
r['preservation_regressions']=['test/test_cpu_array_copy.py','test/test_cpu_match_cache.py','test/test_cpu_psd_variation.py','test/test_cpu_skymax_chisq.py','test/test_cpu_strain_cache.py','test/test_fft_cpu_preservation.py','test/test_scheme_runtime.py']
r['scope']='Main conversion plus separate optional followups. PR16 retains its pre-existing standalone general FFT batch changes; it is outside the main conversion and main qualification. PR17 retains its pre-existing optional native changes, byte-identical to the previously published PR17.'
(O/'source-preservation.json').write_text(json.dumps(r,indent=2)+'\n');print('PASS original ancestor/no PR20/restoration leaves native unchanged all15; main3CPUbackend files byte-exact; setup AST unchanged except Torch extra')
