import copy,hashlib,json,pathlib,subprocess
U=pathlib.Path(__file__).parent
T=pathlib.Path('/private/tmp/pycbc-torch-publication-optional-20260908')
NEW=json.loads((U.parent/'final-heads.json').read_text())['15']
assert NEW=='9d4e4f6905d9f109d559284326d62173a854264f'
ALLOW='test/test_array_lal.py'
def git(*args):return subprocess.check_output(['git',*args],cwd=T,text=True).strip()
def write(name,data):(U/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
def diff(a,b):return subprocess.check_output(['git','diff','--binary',a,b],cwd=T)
prior=json.loads((U/'final-heads.json').read_text())
OLD=prior['base_main_pr15']
assert OLD=='5d09a44b88d6c4e4717229897cba1101b7b2e8db'
assert git('status','--porcelain')==''
assert git('diff','--name-only',OLD,NEW)==ALLOW
main_diff=diff(OLD,NEW)
for n,b in prior['branches'].items():assert git('rev-parse',b['ref'])==b['head']
write('final-heads-docs-rebased.json',prior)
heads={};parents={};records={};fingerprints={}
for n in ['19','16','17']:
    b=prior['branches'][n]
    parent=heads['19'] if n=='16' else NEW
    subprocess.run(['git','checkout','--detach',parent],cwd=T,check=True)
    subprocess.run(['git','cherry-pick',b['head']],cwd=T,check=True)
    head=git('rev-parse','HEAD')
    assert git('rev-parse',head+'^')==parent
    assert diff(b['head'],head)==main_diff
    assert diff(parent,head)==(U/b['final_diff']).read_bytes()
    tested=b['tested_head']
    assert diff(tested,head)==diff(prior['tested_base_main_pr15'],NEW)
    fp=json.loads((U/b['source_fingerprint']['manifest']).read_text())
    entries=git('ls-tree','-r',head,'--',*fp['path_scope']).splitlines()
    def filtered(rows):return [s for s in rows if s.split('\t',1)[1]!=ALLOW]
    assert filtered(entries)==filtered(fp['entries'])
    old_test=next(s for s in fp['entries'] if s.split('\t',1)[1]==ALLOW)
    new_test=next(s for s in entries if s.split('\t',1)[1]==ALLOW)
    assert old_test!=new_test
    assert git('rev-parse',head+':'+ALLOW)==git('rev-parse',NEW+':'+ALLOW)
    digest=hashlib.sha256(('\n'.join(entries)+'\n').encode()).hexdigest()
    fpname=f'pr{n}-final-source-fingerprint.json'
    write(fpname,{'head':head,'tested_head':tested,'path_scope':fp['path_scope'],'git_ls_tree_sha256':digest,'entries':entries,'comparison':'Matches tested fingerprint except the inherited shared test_array_lal.py unittest.skipIf correction; runtime and every other test are byte-identical.','shared_test_exception':ALLOW})
    fingerprints[n]={'head':head,'manifest':fpname,'git_ls_tree_sha256':digest}
    heads[n]=head;parents[n]=parent
    records[n]={'tested_head':tested,'tested_parent':b['tested_parent'],'prior_docs_rebased_head':b['head'],'prior_docs_rebased_parent':b['parent'],'rebased_head':head,'rebased_parent':parent,'changed_paths_from_prior_docs_rebase':[ALLOW],'changed_paths_from_tested':git('diff','--name-only',tested,head).splitlines(),'only_identical_inherited_main_changes':True,'feature_diff_byte_identical':True,'runtime_and_all_other_tests_byte_identical':True,'shared_test_exception':ALLOW,'shared_test_identical_to_final_main':True,'tested_test_blob':old_test,'final_test_blob':new_test,'final_source_fingerprint':fingerprints[n],'test_rerun':False,'test_reuse_reason':'All changes from original tested head exactly equal main documentation updates plus the inherited test_array_lal.py skip fix; optional feature diffs and all remaining runtime/test/build/tool/CI bytes remain identical.'}
transaction='start\n'+''.join('update '+prior['branches'][n]['ref']+' '+heads[n]+' '+prior['branches'][n]['head']+'\n' for n in ['19','16','17'])+'prepare\ncommit\n'
subprocess.run(['git','update-ref','--stdin'],input=transaction,text=True,cwd=T,check=True)
subprocess.run(['git','checkout','codex/publication-20260908-pr17'],cwd=T,check=True)
assert git('status','--porcelain')==''
final=copy.deepcopy(prior)
final['base_main_pr15']=NEW
final['prior_docs_rebased_main_pr15']=OLD
final['final_rebase_receipt']='final-rebase-verification.json'
final['transplant_validation']='Completed onto 9d4e4f6905. Optional feature diffs and all tested runtime/test/build/tool/CI bytes are identical except the inherited shared test_array_lal.py unittest.skipIf correction. Original receipts retain actual executed heads; the parent separately verified the corrected LAL test.'
for n,b in final['branches'].items():
    b['prior_docs_rebased_head']=b['head'];b['prior_docs_rebased_parent']=b['parent']
    b['head']=heads[n];b['parent']=parents[n]
    b.pop('validation_reused_after_docs_only_rebase',None)
    b['validation_reused_after_inherited_docs_and_test_skip_fix']=True
    b['final_source_fingerprint']=fingerprints[n]
    b['shared_test_exception']=ALLOW
write('final-heads.json',final)
parent_verification={'source':'Parent user message; not executed by this optional-branch worker','reported':'1,288 array legacy tests passed with one LAL skip at the updated PR5; additional PR8 14-file suite passed.','shared_fix':'Unsupported Torch LAL conversion now uses unittest.skipIf instead of simple_exit(str).'}
write('final-rebase-verification.json',{'state':'pass','old_main':OLD,'new_main':NEW,'main_changed_paths':[ALLOW],'branches':records,'parent_reported_shared_fix_verification':parent_verification,'publication_performed':False,'outstanding_jobs':[]})
write('final-status.json',{'head':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),'status':git('status','--porcelain'),'outstanding_jobs':[]})
summary=json.loads((U/'validation-summary.json').read_text())
summary['current_main']=NEW
summary['final_rebase_verification']='final-rebase-verification.json'
summary['shared_test_exception']=ALLOW
for n,b in summary['branches'].items():b['rebased_head']=heads[n];b['rebased_parent']=parents[n]
write('validation-summary.json',summary)
review=(U/'REVIEW.md').read_text()
review=review.replace('on main #15 at `'+OLD+'`','on main #15 at `'+NEW+'`')
review=review.replace('Validation on byte-identical tested source','Validation before inherited documentation / LAL-test fix')
for n in ['19','16','17']:
    b=prior['branches'][n]
    review=review.replace('| #'+n+' | `'+b['head']+'` | `'+b['parent']+'` |','| #'+n+' | `'+heads[n]+'` | `'+parents[n]+'` |')
start=review.index('The three branches were transplanted onto main `')
review=review[:start]+'The three branches were finally transplanted onto main `'+NEW+'`. Relative to each original tested head, only `docs/torch_followups.rst`, `docs/torch_search.rst` and the shared `test/test_array_lal.py` skip correction differ, exactly matching the inherited main changes. Every optional feature diff is byte-identical. Runtime, build, tool, CI and all other test files are byte-identical. The exception is the parent’s replacement of unsupported Torch `simple_exit(str)` with `unittest.skipIf`; the parent reports 1,288 array legacy tests passing with one LAL skip and a passing additional 14-file PR8 suite.\n\n`final-rebase-verification.json` and `final-heads.json` retain original tested, intermediate documentation-rebased and final identities. Original stage receipts and tested-source fingerprints remain unchanged; final fingerprints are separate `prNN-final-source-fingerprint.json` files. Full optional suites were not rerun, as requested. No optional runtime/test fixes were added.\n'
(U/'REVIEW.md').write_text(review)
write('active.json',{'state':'complete','outstanding_jobs':[],'final_heads':'final-heads.json','final_rebase_verification':'final-rebase-verification.json'})
files=[p for p in U.iterdir() if p.is_file() and p.name not in ['SHA256SUMS','active.json']]
(U/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in sorted(files)))
print(json.dumps({'state':'pass','main':NEW,'heads':heads,'parents':parents,'runtime_and_tests_identical_except_shared_LAL_skip':True,'tracked_status':git('status','--porcelain')},indent=2))
