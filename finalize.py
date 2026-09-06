"""Seal completed benchmark outputs and verify the measured checkouts."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/home/xangma/pycbc-torch-benchmark-20260906')
PY = '/home/xangma/pycbc-torch-split-20260905/venv/bin/python'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert json.loads((ROOT/'live/status.json').read_text())['finished']
    assert json.loads((ROOT/'supplement/status.json').read_text())['finished']
    sources = []
    for row in json.loads((ROOT/'preparation.json').read_text()):
        repo = ROOT/row['name']
        sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
        status = subprocess.check_output(['git','status','--porcelain','--untracked-files=normal'],cwd=repo,text=True)
        assert sha == row['sha'] and not status, (sha, status)
        changed = [name for name, expected in row['binaries'].items()
                   if digest(repo/name) != expected]
        assert not changed, changed
        sources.append(dict(name=row['name'],sha=sha,clean=True,
                            native_binaries_unchanged=len(row['binaries'])))
    packages = subprocess.check_output([PY,'-m','pip','--disable-pip-version-check',
                                        'list','--format=json'],text=True)
    (ROOT/'packages.json').write_text(packages)
    classifications = {}
    for part in ['fft','probes','waveform','inference']:
        rows = []
        for p in sorted((ROOT/'supplement'/part).glob('*.json')):
            d = json.loads(p.read_text())
            if 'status' in d:
                rows.append(dict(file=p.name,status=d['status']))
        classifications[part] = rows
    result = dict(sealed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  sources=sources,classifications=classifications)
    files = []
    for directory in ['live','smoke','supplement','logs']:
        for p in sorted((ROOT/directory).rglob('*')):
            if p.is_file():
                files.append(p)
    for name in ['preparation.json','packages.json','prepare.py','run-live.py',
                 'run-smoke.py','run-supplement.py','finalize.py']:
        files.append(ROOT/name)
    for p in sorted((ROOT/'harness').rglob('*.py')):
        if 'smoke' not in p.parts:
            files.append(p)
    result['files'] = {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size,sha256=digest(p))
                       for p in files}
    (ROOT/'sealed-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(sources=sources,files=len(files),
                         statuses={key:{status:sum(x['status']==status for x in rows)
                                        for status in sorted({x['status'] for x in rows})}
                                   for key,rows in classifications.items()})))


if __name__ == '__main__':
    main()
