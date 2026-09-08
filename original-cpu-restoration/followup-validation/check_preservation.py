import ast,json,pathlib,shutil,subprocess
out=pathlib.Path(__file__).parent
root='/Users/xangma/repos/pycbc'
main='aa6b795a63bb18c4e63e4f4c203ca6e7c039d0f0'
original='40e94792b3'
def blob(ref,path):
    return subprocess.check_output(['git','show',f'{ref}:{path}'],cwd=root)
def units(data):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(data).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
summary={'main_head':main,'original':subprocess.check_output(['git','rev-parse',original],cwd=root,text=True).strip(),'main_byte_preserved':{p:blob(main,p)==blob(original,p) for p in ['pycbc/fft/fftw.py','pycbc/fft/npfft.py','pycbc/fft/mkl.py']},'prs':{}}
for pr in (16,17):
    wt=pathlib.Path(f'/private/tmp/pycbc-original-cpu-followup-validation-20260908-pr{pr}')
    reference=units(blob(original,'pycbc/fft/fftw.py'))
    current=units((wt/'pycbc/fft/fftw.py').read_bytes())
    changed=[n for n in reference if reference[n]!=current.get(n)]
    assert changed == (['insert_fft_options','verify_fft_options'] if pr==16 else [])
    checks={}
    for label,args in [('lint-modules',['/Users/xangma/miniconda3/bin/flake8','pycbc/','test/','--select','F401','--exclude','__init__.py,version.py,test_schemes.py']),('lint-bin',['/Users/xangma/miniconda3/bin/flake8','--select','F401']+[str(p.relative_to(wt)) for p in (wt/'bin').rglob('pycbc_*') if p.is_file()]),('diff-check',['git','diff','--check'])]:
        result=subprocess.run(args,cwd=wt,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (out/f'pr{pr}-{label}.log').write_text(result.stdout)
        checks[label]={'exit_code':result.returncode,'command':args}
    summary['prs'][str(pr)]={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=wt,text=True).strip(),'fftw_changed_top_level_units':changed,'fftw_preserved_top_level_units':len(reference)-len(changed),'checks':checks,'qlty_available':shutil.which('qlty')}
assert all(summary['main_byte_preserved'].values())
(out/'preservation-and-quality.json').write_text(json.dumps(summary,indent=2)+'\n')
print('Main FFTW/NumPy/MKL byte-preserved:', summary['main_byte_preserved'])
for pr,result in summary['prs'].items():
    print(pr,result['fftw_preserved_top_level_units'],'original FFTW functions/classes;', {label:check['exit_code'] for label,check in result['checks'].items()})
