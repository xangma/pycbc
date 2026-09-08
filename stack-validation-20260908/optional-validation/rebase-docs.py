import copy,hashlib,json,pathlib,subprocess
U=pathlib.Path(__file__).parent
T=pathlib.Path('/private/tmp/pycbc-torch-publication-optional-20260908')
NEW='5d09a44b88d6c4e4717229897cba1101b7b2e8db'
def git(*args):return subprocess.check_output(['git',*args],cwd=T,text=True).strip()
def write(name,data):(U/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
def blobdiff(a,b):return subprocess.check_output(['git','diff','--binary',a,b],cwd=T)
prior=json.loads((U/'final-heads.json').read_text())
OLD=prior['base_main_pr15']
assert OLD=='542f7f3dc54db454128ef509b4e936f94a5f8072'
assert git('status','--porcelain')==''
assert subprocess.call(['git','merge-base','--is-ancestor',OLD,NEW],cwd=T)==0
doc_paths=git('diff','--name-only',OLD,NEW).splitlines()
assert doc_paths==['docs/torch_followups.rst','docs/torch_search.rst'],doc_paths
for n,b in prior['branches'].items():assert git('rev-parse',b['ref'])==b['head']
write('final-heads-tested.json',prior)
(U/'SHA256SUMS-tested').write_bytes((U/'SHA256SUMS').read_bytes())
heads={};parents={};records={}
for n in ['19','16','17']:
    b=prior['branches'][n]
    parent=heads['19'] if n=='16' else NEW
    subprocess.run(['git','checkout','--detach',parent],cwd=T,check=True)
    subprocess.run(['git','cherry-pick',b['head']],cwd=T,check=True)
    head=git('rev-parse','HEAD')
    assert git('rev-parse',head+'^')==parent
    assert blobdiff(b['head'],head)==blobdiff(OLD,NEW)
    assert blobdiff(parent,head)==(U/b['final_diff']).read_bytes()
    fp=json.loads((U/b['source_fingerprint']['manifest']).read_text())
    entries=git('ls-tree','-r',head,'--',*fp['path_scope'])
    digest=hashlib.sha256((entries+'\n').encode()).hexdigest()
    assert entries.splitlines()==fp['entries']
    assert digest==fp['git_ls_tree_sha256']
    heads[n]=head;parents[n]=parent
    records[n]={'tested_head':b['head'],'tested_parent':b['parent'],'rebased_head':head,'rebased_parent':parent,'changed_paths_from_tested':doc_paths,'only_identical_main_docs_change':True,'feature_diff_byte_identical':True,'runtime_test_build_tools_ci_fingerprint_identical':True,'source_fingerprint_sha256':digest,'source_fingerprint_entries':len(fp['entries']),'test_rerun':False,'test_reuse_reason':'All tracked differences are precisely the same two documentation changes as main; the per-PR feature diff and tested source fingerprints remain byte-identical.'}
# Compare-and-swap all owned branch refs only, after all transplants verify.
transaction='start\n'+''.join('update '+prior['branches'][n]['ref']+' '+heads[n]+' '+prior['branches'][n]['head']+'\n' for n in ['19','16','17'])+'prepare\ncommit\n'
subprocess.run(['git','update-ref','--stdin'],input=transaction,text=True,cwd=T,check=True)
subprocess.run(['git','checkout','codex/publication-20260908-pr17'],cwd=T,check=True)
assert git('status','--porcelain')==''
final=copy.deepcopy(prior)
final['tested_base_main_pr15']=OLD
final['base_main_pr15']=NEW
final['docs_rebase_receipt']='docs-rebase-verification.json'
final['transplant_validation']='Completed onto 5d09a44b88. Full tracked-file differences match the two main documentation changes exactly. Per-PR feature diffs and runtime/test/build/tool/CI fingerprints are byte-identical. Original test receipts retain their executed heads.'
final.pop('transplant_order',None)
for n,b in final['branches'].items():
    b['tested_head']=b['head'];b['tested_parent']=b['parent']
    b['head']=heads[n];b['parent']=parents[n]
    b['validation_reused_after_docs_only_rebase']=True
write('final-heads.json',final)
write('docs-rebase-verification.json',{'state':'pass','old_main':OLD,'new_main':NEW,'main_changed_paths':doc_paths,'branches':records,'publication_performed':False,'outstanding_jobs':[]})
write('final-status.json',{'head':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),'status':git('status','--porcelain'),'outstanding_jobs':[]})
summary=json.loads((U/'validation-summary.json').read_text())
summary['tested_main']=summary.pop('main')
summary['current_main']=NEW
summary['docs_rebase_verification']='docs-rebase-verification.json'
for n,b in summary['branches'].items():b['rebased_head']=heads[n];b['rebased_parent']=parents[n]
write('validation-summary.json',summary)
review=(U/'REVIEW.md').read_text()
review=review.replace('on main #15 at `'+OLD+'`','on main #15 at `'+NEW+'`')
review=review.replace('| PR | Head | Parent | Validation |','| PR | Rebased head | Rebased parent | Validation on byte-identical tested source |')
for n in ['19','16','17']:
    b=prior['branches'][n]
    review=review.replace('| #'+n+' | `'+b['head']+'` | `'+b['parent']+'` |','| #'+n+' | `'+heads[n]+'` | `'+parents[n]+'` |')
start=review.index('For a later docs-only main commit,')
review=review[:start]+'The three branches were transplanted onto main `'+NEW+'` after validation. Only `docs/torch_followups.rst` and `docs/torch_search.rst` differ from each tested head, exactly matching the main documentation change. Every per-PR feature diff and runtime/test/build/tool/CI fingerprint is byte-identical. `docs-rebase-verification.json` and `final-heads.json` retain both tested and rebased identities; the original stage receipts and fingerprints are unchanged. Tests were not repeated after this documentation-only transplant.\n'
(U/'REVIEW.md').write_text(review)
write('active.json',{'state':'complete','outstanding_jobs':[],'final_heads':'final-heads.json','docs_rebase_verification':'docs-rebase-verification.json'})
files=[p for p in U.iterdir() if p.is_file() and p.name not in ['SHA256SUMS','active.json']]
(U/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in sorted(files)))
print(json.dumps({'state':'pass','main':NEW,'heads':heads,'parents':parents,'all_source_and_feature_diffs_identical':True,'tracked_status':git('status','--porcelain')},indent=2))
