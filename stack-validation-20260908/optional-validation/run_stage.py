import hashlib,json,os,pathlib,socket,subprocess,sys,time
U=pathlib.Path(__file__).parent
TREE=pathlib.Path('/private/tmp/pycbc-torch-publication-optional-20260908')
PY=str(U/'venv/bin/python')
QLTY=str(U.parent/'qlty-runtime/qlty-aarch64-apple-darwin/qlty')
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYCBC_TEST_SCHEME='torch:cpu',PYTHONPATH=str(TREE),PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(U/'tmp'),PIP_DISABLE_PIP_VERSION_CHECK='1',QLTY_TELEMETRY='off',MPLCONFIGDIR=str(U/'mpl-cache'))
label,stage=sys.argv[1:3]
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TREE,text=True).strip()
parent=subprocess.check_output(['git','rev-parse','HEAD^'],cwd=TREE,text=True).strip()
if stage=='build':
 cmd=[PY,'-m','pip','install','--no-index','--no-deps','--no-build-isolation','--no-cache-dir','-e',str(TREE)]
elif stage=='units':
 old={'pr19':'format-fft','pr16':'fft-followup','pr17':'cpu-followup'}[label.split('-')[0]]
 record=json.loads((pathlib.Path('/Users/xangma/repos/pycbc/artifacts/torch-performance-fix-20260906/quality-final-v6-r2')/(old+'-units-result.json')).read_text())
 tests=[p for p in record['command'] if p.startswith('test/')]
 tests+=['test/test_fft.py','test/test_torch_frame_loader.py','test/test_frame.py']
 tests=list(dict.fromkeys(tests))
 tests=[p for p in tests if (TREE/p).exists()]
 cmd=[PY,'-m','pytest','-q','-ra','-p','no:cacheprovider','--tb=short','-k','not cuda and not mps','--basetemp='+str(U/(label+'-pytest-tmp')),'--junitxml='+str(U/(label+'-units.xml')),*tests]
elif stage=='f401':
 files=sorted([str(p.relative_to(TREE)) for p in (TREE/'bin').rglob('pycbc_*') if p.is_file()]+[str(p.relative_to(TREE)) for p in (TREE/'pycbc').rglob('*.py') if '__init__' not in str(p) and 'version.py' not in str(p)]+[str(p.relative_to(TREE)) for p in (TREE/'test').rglob('*.py') if 'test_schemes' not in str(p)])
 cmd=[PY,'-m','flake8','--select=F401',*files]
elif stage=='qlty':
 parent=sys.argv[3]
 cmd=[QLTY,'check','--no-upgrade-check','--no-fix','--sarif','--no-progress','--skip-source-fetch','--no-cache','--upstream',parent]
else: raise ValueError(stage)
log=U/(label+'-'+stage+('.sarif' if stage=='qlty' else '.log'))
err=U/(label+'-'+stage+'-stderr.log')
started=time.time()
with log.open('w') as out,err.open('w') as stderr:
 child=subprocess.Popen(cmd,cwd=TREE,env=env,stdout=out,stderr=stderr,start_new_session=True)
 live=dict(label=label,stage=stage,host=socket.gethostname(),cwd=str(TREE),command=cmd,head=head,parent=parent,started=started,pid=child.pid,controller_pid=os.getpid(),log=str(log),stderr=str(err),stop_command=f'kill -TERM -- -{child.pid}',environment={k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PYCBC_TEST_SCHEME','PYTHONPATH','PYTHONDONTWRITEBYTECODE','TMPDIR']})
 (U/'active.json').write_text(json.dumps(live,indent=2)+'\n')
 print(json.dumps({k:v for k,v in live.items() if k not in ('command','environment')}),flush=True)
 rc=child.wait()
receipt=dict(**live,returncode=rc,finished=time.time(),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),stderr_sha256=hashlib.sha256(err.read_bytes()).hexdigest())
(U/(label+'-'+stage+'-result.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print('FINISHED',label,stage,rc,round(time.time()-started,2),flush=True)
sys.exit(rc)
