from pathlib import Path
import ast, inspect, json, os, socket, subprocess, time
U=Path(__file__).resolve().parent
T=Path('/private/tmp/pycbc-torch-publication-20260908')
PY='/private/tmp/pycbc-torch-publication-20260908-venv/bin/python'
heads=json.loads((U/'final-heads.json').read_text())
optional=json.loads((U/'optional-validation/final-heads.json').read_text())
assert optional['base_main_pr15']==heads['15']
for k,v in optional['branches'].items():
    assert subprocess.check_output(['git','rev-parse',v['ref']],cwd=T,text=True).strip()==v['head']
    actual=subprocess.check_output(['git','diff',v['parent'],v['head']],cwd=T)
    assert actual==(U/'optional-validation'/v['final_diff']).read_bytes()
    heads[k]=v['head']
(U/'final-heads.json').write_text(json.dumps(heads,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(T),PYCBC_TEST_SCHEME='cpu',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
def run(name,command):
    log=U/(name+'.log')
    with log.open('w') as f:
        p=subprocess.Popen(command,cwd=T,env=env,stdout=f,stderr=subprocess.STDOUT)
        receipt=dict(host=socket.gethostname(),cwd=str(T),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=T,text=True).strip(),command=command,pid=p.pid,log=str(log),started=time.time())
        print(json.dumps(receipt),flush=True)
        receipt.update(returncode=p.wait(),finished=time.time())
    (U/(name+'-result.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    if receipt['returncode']:raise RuntimeError(name+' failed')
try:
    subprocess.run(['git','switch','--detach',heads['18']],cwd=T,check=True)
    run('format-pr18-frame-standalone',[PY,'test/test_frame.py'])
finally:
    subprocess.run(['git','switch','--detach',heads['15']],cwd=T,check=True)
run('lal-cpu-conversion',[PY,'-m','pytest','-q','-p','no:cacheprovider','--junitxml='+str(U/'lal-cpu-conversion.xml'),'test/test_array_lal.py'])
def normalized(source):
    tree=ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
            node.body[0].value.value=inspect.cleandoc(node.body[0].value.value).strip()
    return ast.dump(tree,include_attributes=False)
production='fad7d8440bfde083f2e94ee62a0017492dfc4013'
different=subprocess.check_output(['git','diff','--name-only',production,heads['15'],'--','pycbc','bin','test'],cwd=T,text=True).splitlines()
formatted=['pycbc/fft/torchfft.py','pycbc/frame/frame.py','pycbc/vetoes/chisq_torch.py']
assert sorted(different)==sorted(formatted+['test/test_array_lal.py']),different
for path in formatted:
    a=subprocess.check_output(['git','show',production+':'+path],cwd=T,text=True)
    b=subprocess.check_output(['git','show',heads['15']+':'+path],cwd=T,text=True)
    assert normalized(a)==normalized(b),path
(U/'final-source-equivalence.json').write_text(json.dumps(dict(state='pass',production_revision=production,assembled_head=heads['15'],only_runtime_differences=formatted,comparison='AST identity after docstring whitespace normalization; all other runtime and executable files byte-identical',test_only_difference='test/test_array_lal.py',test_validation=['additional-pr5-tests.xml','lal-cpu-conversion.xml']),indent=2)+'\n')
print('Remaining local checks and final source mapping PASS')
