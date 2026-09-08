"""Read-only source inventory and outgoing text scan; write only beside this file."""
import hashlib
import json
from pathlib import Path
import re
import tarfile

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parent.parent
ROOTS = [
    'torch-fft-optimization-20260908', 'torch-residual-optimization-20260908',
    'torch-offline-cuda-graph-20260908', 'torch-cpu-workspace-policy-20260908',
    'torch-profiling-investigation-20260908',
]
HEX = re.compile(r'[a-f0-9]{64}')
PATTERNS = {
    'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    'service_token': r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,}|xox[baprs]-[A-Za-z0-9-]{20,}|AKIA[A-Z0-9]{16})\b',
    'jwt': r'\beyJ[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b',
    'credential_url': r'https?://[^\s/:@]+:[^\s/@]+@',
    'authorization': r'(?i)(?:authorization[\s\x22\x27:=]+(?:bearer|basic)\s+[A-Za-z0-9+/=_-]{12,})',
    'assigned_secret': r'(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[\x22\x27]?\s*[:=]\s*[\x22\x27][^\x22\x27\n]{8,}[\x22\x27]',
    'personal_data': r'(?i)(?:\b(?:social security|credit card number|date of birth|home address)\b)',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def scan(data, name):
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        return False, []
    if '\0' in text:
        return False, []
    hits = []
    for rule, pattern in PATTERNS.items():
        for match in re.finditer(pattern, text):
            hits.append(dict(path=name, rule=rule, line=text.count('\n', 0, match.start()) + 1))
    return True, hits


def snapshot():
    report = {'roots': {}, 'scan': {'text_files': 0, 'text_bytes': 0, 'findings': []}}
    for name in ROOTS:
        root = ARTIFACTS / name
        files, omitted = {}, []
        for p in sorted(root.rglob('*')):
            if p.is_symlink():
                raise ValueError('Unexpected symlink: ' + str(p))
            if not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            data = p.read_bytes()
            record = dict(bytes=len(data), sha256=digest(data), mode=p.stat().st_mode & 0o777)
            if '__pycache__' in p.parts or p.suffix in {'.pyc', '.pyo'}:
                omitted.append(dict(path=rel, reason='Interpreter cache; not evidence.', **record))
                continue
            files[rel] = record
            is_text, hits = scan(data, name + '/' + rel)
            report['scan']['text_files'] += is_text
            report['scan']['text_bytes'] += len(data) if is_text else 0
            report['scan']['findings'].extend(hits)
        seal = root / 'SHA256SUMS'
        expected = {}
        if seal.exists():
            for line in seal.read_text().splitlines():
                sha, rel = line.split('  ', 1)
                if not HEX.fullmatch(sha) or rel in expected:
                    raise ValueError('Invalid seal ' + name)
                expected[rel] = sha
                if files.get(rel, {}).get('sha256') != sha:
                    raise ValueError('Seal mismatch ' + name + '/' + rel)
        archives = {}
        for rel in files:
            if not rel.endswith(('.tar', '.tar.gz', '.tgz')):
                continue
            members = []
            with tarfile.open(root / rel, 'r:*') as archive:
                seen = set()
                for m in archive.getmembers():
                    if not (m.isfile() or m.isdir()) or m.name in seen:
                        raise ValueError('Unexpected tar member ' + rel + ':' + m.name)
                    seen.add(m.name)
                    row = dict(name=m.name, type='file' if m.isfile() else 'directory', bytes=m.size,
                               mode=m.mode, uid=m.uid, gid=m.gid, mtime=m.mtime, pax_headers=m.pax_headers)
                    if m.isfile():
                        data = archive.extractfile(m).read()
                        row['sha256'] = digest(data)
                        is_text, hits = scan(data, name + '/' + rel + '::' + m.name)
                        report['scan']['text_files'] += is_text
                        report['scan']['text_bytes'] += len(data) if is_text else 0
                        report['scan']['findings'].extend(hits)
                    members.append(row)
            archives[rel] = dict(**files[rel], members=members)
        manifests = []
        for rel in files:
            if not rel.endswith('.json') or not any(x in Path(rel).name for x in ['manifest', 'pins', 'source']):
                continue
            try:
                obj = json.loads((root / rel).read_text())
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(obj, dict):
                continue
            refs = []
            for key, val in obj.items():
                sha = val if isinstance(val, str) else val.get('sha256') if isinstance(val, dict) else None
                if not isinstance(sha, str) or not HEX.fullmatch(sha):
                    continue
                candidate = (Path(rel).parent / key).as_posix()
                exact = files.get(candidate, {}).get('sha256') == sha
                matches = [f for f, meta in files.items() if meta['sha256'] == sha]
                refs.append(dict(reference=key, sha256=sha, status='relative_match' if exact else 'content_match' if matches else 'external_or_historical', selected_matches=matches))
            if refs:
                manifests.append(dict(path=rel, sha256=files[rel]['sha256'], entries=refs))
        report['roots'][name] = dict(source_root=str(root), files=files, omitted=omitted,
            original_seal=dict(present=seal.exists(), entries=len(expected), sha256=digest(seal.read_bytes()) if seal.exists() else None,
                               all_entries_verified=True if seal.exists() else None,
                               unlisted_files=sorted(set(files)-set(expected)-{'SHA256SUMS'})),
            archives=archives, authoritative_inventories=manifests)
    return report


if __name__ == '__main__':
    result = snapshot()
    (HERE / 'input-inspection.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: dict(files=len(v['files']), bytes=sum(f['bytes'] for f in v['files'].values()),
                             archives=len(v['archives']), seal=v['original_seal'], omitted=len(v['omitted']))
                      for k,v in result['roots'].items()}, indent=2))
    print(json.dumps(result['scan'], indent=2))
