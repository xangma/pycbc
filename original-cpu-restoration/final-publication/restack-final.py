"""Rebuild the verified restoration with owner-scoped formatting, documentation and optional followup fixes."""
import json
from pathlib import Path
import subprocess

O=Path(__file__).resolve().parent
W=Path('/private/tmp/pycbc-original-cpu-restoration-final-20260908')
R=Path('/Users/xangma/repos/pycbc')
BASE='40e94792b3edf59f39b18b65102b28a4f74433a7'
def git(*a):
    r=subprocess.run(['git',*a],cwd=W if W.exists() else R,text=True,capture_output=True)
    with (O/'replay-final.log').open('a') as f:f.write('$ git '+' '.join(a)+'\n'+r.stdout+r.stderr)
    r.check_returncode()
    return r.stdout.strip()
def main():
    source=json.loads((O/'manifest-v2.json').read_text())
    path=O/'manifest-final.json'
    out=json.loads(path.read_text()) if path.exists() else {'cpu_base':BASE,'status':'replay_running','prs':[]}
    done={r['pr']:r for r in out['prs']}
    fixes=json.loads((O/'owner-fixes-final.json').read_text())
    if not W.exists():git('worktree','add','--detach',str(W),BASE)
    for row in source['prs']:
        n=row['pr']
        if n in done:continue
        if str(n) not in fixes:break
        newbase=done[row['parent_pr']]['new_head'] if row['parent_pr'] else BASE
        branch=f'codex/original-cpu-restoration-final-20260908-pr{n:02d}'
        exists=subprocess.run(['git','show-ref','--verify','--quiet','refs/heads/'+branch],cwd=W).returncode==0
        git('checkout',branch) if exists else git('checkout','-b',branch,newbase)
        picked=[]
        for commit in git('rev-list','--reverse',row['new_base']+'..'+row['new_head']).splitlines()+fixes[str(n)]:
            prior=git('log','--format=%H','--fixed-strings','--grep=cherry picked from commit '+commit,newbase+'..HEAD')
            if not prior:
                git('cherry-pick','-x',commit)
                prior=git('rev-parse','HEAD')
            picked.append({'source':commit,'new':prior})
        new=dict(row,new_base=newbase,new_head=git('rev-parse','HEAD'),staging_ref=branch,restoration_commits=picked)
        out['prs'].append(new);done[n]=new;path.write_text(json.dumps(out,indent=2)+'\n');print(n,new['new_head'],flush=True)
    out['status']='replay_complete' if len(done)==15 else 'awaiting_owner_fixes'
    path.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
