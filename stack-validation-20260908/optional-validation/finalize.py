import hashlib,json,pathlib,re,shutil,subprocess,xml.etree.ElementTree as ET
U=pathlib.Path(__file__).parent
TREE=pathlib.Path('/private/tmp/pycbc-torch-publication-optional-20260908')
BASE='542f7f3dc54db454128ef509b4e936f94a5f8072'
def git(*args):return subprocess.check_output(['git',*args],cwd=TREE,text=True).strip()
def write(name,data):(U/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
assert git('status','--porcelain','--untracked-files=no')==''
originals={'19':'03494ad311e03fbc1255e180ea130cefa3db55b2','16':'3d15380f836f9fe17d1d317837465693335a712d','17':'3e88711c1453d2ffab5eca42ec3207d34c50ddef'}
heads={n:git('rev-parse','codex/publication-20260908-pr'+n) for n in originals}
parents={'19':BASE,'16':heads['19'],'17':BASE}
results={};branches={};fingerprints={}
for n,head in heads.items():
    parent=parents[n]
    assert git('rev-parse',head+'^')==parent
    build=json.loads((U/f'pr{n}-build-result.json').read_text())
    tests=json.loads((U/f'pr{n}-units-result.json').read_text())
    f401=json.loads((U/f'pr{n}-f401-result.json').read_text())
    qlty=json.loads((U/f'pr{n}-qlty-result.json').read_text())
    for r in [build,tests,f401,qlty]:assert r['returncode']==0 and r['head']==head
    assert qlty['parent']==parent
    tail=(U/f'pr{n}-units.log').read_text()
    summary=re.findall(r'^\d+ passed,.*$',tail,re.M)[-1]
    xml=ET.parse(U/f'pr{n}-units.xml').getroot()
    suites=list(xml.iter('testsuite'))
    assert sum(int(s.get('failures','0'))+int(s.get('errors','0')) for s in suites)==0
    sarif=json.loads((U/f'pr{n}-qlty.sarif').read_text())
    findings=sum(len(r.get('results',[])) for r in sarif['runs'])
    assert findings==0
    (U/f'pr{n}-final.diff').write_text(subprocess.check_output(['git','diff','--binary',parent,head],cwd=TREE,text=True))
    source_paths=['pycbc','bin','test','tools','setup.py','setup.cfg','pyproject.toml','.github/workflows']
    tree=git('ls-tree','-r',head,'--',*source_paths)
    fingerprint={'head':head,'path_scope':source_paths,'git_ls_tree_sha256':hashlib.sha256((tree+'\n').encode()).hexdigest(),'entries':tree.splitlines(),'comparison':'After a docs-only transplant, identical mode/blob/path records prove these runtime, test, build, tool and CI files retain identical bytes.'}
    write(f'pr{n}-source-fingerprint.json',fingerprint)
    fingerprints[n]={'manifest':f'pr{n}-source-fingerprint.json','git_ls_tree_sha256':fingerprint['git_ls_tree_sha256']}
    results[n]={'head':head,'parent':parent,'state':'pass','pytest_summary':summary,'test_file_count':sum(x.startswith('test/') for x in tests['command']),'unit_log':f'pr{n}-units.log','unit_xml':f'pr{n}-units.xml','build_receipt':f'pr{n}-build-result.json','unit_receipt':f'pr{n}-units-result.json','f401_receipt':f'pr{n}-f401-result.json','qlty_receipt':f'pr{n}-qlty-result.json','qlty_sarif':f'pr{n}-qlty.sarif','qlty_findings':findings,'review':f'pr{n}-review.json'}
    branches[n]={'ref':'refs/heads/codex/publication-20260908-pr'+n,'head':head,'parent':parent,'original_feature_commit':originals[n],'final_diff':f'pr{n}-final.diff','final_diff_sha256':hashlib.sha256((U/f'pr{n}-final.diff').read_bytes()).hexdigest(),'source_fingerprint':fingerprints[n],'validation':results[n]}
write('validation-summary.json',{'state':'pass','main':BASE,'cpu_only':True,'new_source_fixes':[],'branches':results})
write('final-heads.json',{'state':'ready_for_parent_publication','base_main_pr15':BASE,'published_archive':'c4bfea522807742388dc8bcddc86473b9c03b047','branches':branches,'tracked_worktree_clean':True,'publication_performed':False,'transplant_order':['Apply branch 19 feature commit onto any subsequent docs-only main.','Apply branch 16 feature commit onto transplanted 19.','Apply branch 17 feature commit independently onto the same docs-only main.'],'transplant_validation':'Compare each branch source fingerprint after transplant. Existing test receipts identify the heads actually tested here; do not relabel them as executed on a later commit.'})
# Preserve only the generated untracked Qlty outputs, keeping the worktree tidy.
qlty_out=U/'qlty-workspace'
qlty_out.mkdir(exist_ok=True)
for name in ['logs','out','plugin_cachedir','results']:
    p=TREE/'.qlty'/name
    if p.exists() or p.is_symlink():shutil.move(str(p),str(qlty_out/name))
write('final-status.json',{'head':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),'status':git('status','--porcelain'),'outstanding_jobs':[]})
readme='''Optional PR validation — 8 September 2026

All three optional branches are ready for parent publication on main #15 at `542f7f3dc54db454128ef509b4e936f94a5f8072`. See `final-heads.json` for full identities, parents, diffs and receipt paths.

| PR | Head | Parent | Validation |
| --- | --- | --- | --- |
'''
for n in ['19','16','17']:
    readme+=f"| #{n} | `{heads[n]}` | `{parents[n]}` | {results[n]['pytest_summary']} |\n"
readme+='''
Each prefix passed its isolated editable build, the full F401 file scope from `.github/workflows/check_code.yml`, and Qlty 0.644.0 against its actual parent with zero findings. Qlty used `--no-upgrade-check --no-fix --sarif --no-progress --skip-source-fetch --no-cache`; Ruff 0.14.6 was already installed. Full commands, timestamps, host, source identities and log hashes are recorded in the per-stage receipts. JUnit XML and stdout/stderr logs are retained.

The test lists came from the public prior PR validation receipts (23 common files plus the relevant optional suites), with current frame regression coverage added. The three prefixes exercised 24, 27 and 29 files respectively. Tests ran with Torch CPU and one-thread settings on macOS arm64 / Python 3.13.9, Torch 2.9.1 and LALSuite 7.26.1. CUDA/MPS-related cases were deselected; five tests skipped for unavailable CUDA or Linux/x86-64 MKL qualification. These receipts do not claim new Linux-native or GPU qualification. Each prefix also passed 56 subtests. Environment provenance and the built native extension hashes are in `environment.json`.

Review: #19 and #16 retain the exact original patch IDs. #17 merged without conflicts; its added and removed lines match the original feature per file exactly, while its patch ID differs because surrounding main runtime context changed. Both original Cython files are byte-identical to the specified feature commit. The newer in-place promoted MKL workspace, scalar normalization contracts, and CUDA host-array completion changes remain inherited from main. The CPU native peak gate remains default-off and ahead of the inherited Torch fallback. No new runtime/test/lint fix was required and no tests were weakened.

The assigned worktree is left clean on #17. Only the three authorized optional refs were rebuilt. No remote, GPU, push, or PR operation was performed. The earlier prepared ref identities are preserved in `initial-state.json`.

For a later docs-only main commit, transplant #19 then #16 in order; transplant #17 independently. Compare each `prNN-source-fingerprint.json` mode/blob/path inventory against the transplanted head to establish identical runtime, tests, build files, tools and CI bytes. Existing receipts retain their actual tested heads. This permits reuse of the runtime evidence when those bytes are unchanged.
'''
(U/'REVIEW.md').write_text(readme)
# Seal delivery records, excluding the reusable environment, test scratch and tool caches.
files=[p for p in U.iterdir() if p.is_file() and p.name not in ['SHA256SUMS','active.json']]
(U/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in sorted(files)))
print(json.dumps({'heads':heads,'results':{n:r['pytest_summary'] for n,r in results.items()},'status':git('status','--porcelain'),'receipts_sha256':hashlib.sha256((U/'SHA256SUMS').read_bytes()).hexdigest()},indent=2))
