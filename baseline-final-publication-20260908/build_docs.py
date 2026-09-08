from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
u=Path(__file__).resolve().parent
w=Path('/private/tmp/pycbc-torch-baseline-final-results-20260908')
a=u/'render-final';a.mkdir(exist_ok=True)
s=a/'docs';s.mkdir(exist_ok=True)
keep=[p.name for p in (w/'docs').glob('torch*.rst')]+['waveform.rst','waveform_plugin.rst','install.rst','install_cuda.rst','install_lalsuite.rst','install_virtualenv.rst','docker.rst']
for n in keep:shutil.copy2(w/'docs'/n,s/n)
for name in ['images','data','_static','_templates']:
 src=w/'docs'/name
 if src.exists():shutil.copytree(src,s/name,dirs_exist_ok=True)
shutil.copytree(w/'examples',a/'examples',dirs_exist_ok=True)
(s/'index.rst').write_text('Torch baseline and proposal benchmark\n=====================================\n\n.. toctree::\n   :maxdepth: 2\n\n   torch\n   waveform\n   waveform_plugin\n   install\n   api\n')
(s/'api.rst').write_text('Referenced API objects\n======================\n\n.. autoclass:: pycbc.scheme.TorchScheme\n\n.. autofunction:: pycbc.waveform.compress.fd_decompress\n\n.. autofunction:: pycbc.waveform.plugin.add_custom_waveform\n\n.. autoclass:: pycbc.types.frequencyseries.FrequencySeries\n')
(s/'conf.py').write_text("from pathlib import Path\nexec(compile(Path("+repr(str(w/'docs/conf.py'))+").read_text(), "+repr(str(w/'docs/conf.py'))+", 'exec'))\nintersphinx_mapping = {}\nhtml_logo = None\n")
env=os.environ.copy();env['SKIP_PYCBC_DOCS_INCLUDE']='1';env['MPLBACKEND']='Agg';env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env['PATH']
command=[sys.executable,'-m','sphinx','-b','html','-E','-a','-W','--keep-going',str(s),str(a/'html')]
log=u/'sphinx-build.log'
with log.open('w') as f:
 p=subprocess.Popen(command,cwd=s,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(json.dumps({'pid':p.pid,'host':'local','cwd':str(s),'command':command,'log':str(log)}),flush=True)
 code=p.wait()
(u/'sphinx-build-result.json').write_text(json.dumps({'command':command,'returncode':code,'pages':keep,'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip(),'finished':time.time(),'scope':'All remaining Torch pages plus actual waveform/plugin/installation dependencies; repository extensions and theme, real plot/command directives; generated targeted API context; excludes unrelated manual and _include generators, no external intersphinx inventory.'},indent=2)+'\n')
(u/'sphinx-source-hashes.json').write_text(json.dumps({name:hashlib.sha256((s/name).read_bytes()).hexdigest() for name in keep},indent=2)+'\n')
print('Sphinx exit',code,flush=True)
sys.exit(code)
