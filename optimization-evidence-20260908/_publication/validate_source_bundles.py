"""Verify existing bundles in disposable local Git object storage; never edit source."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

from inspect_inputs import HERE, digest, scan
from build_packages import bundle_header


def main():
    inspection = json.loads((HERE / 'input-inspection.json').read_text())
    name = 'torch-fft-optimization-20260908'
    record = inspection['roots'][name]
    source = Path(record['source_root'])
    repo = HERE.parent.parent.parent
    common = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', '--git-common-dir'], text=True).strip()
    objects = (repo / common / 'objects').resolve()
    unique = {}
    for rel, item in record['files'].items():
        if rel.endswith('.bundle'):
            unique.setdefault(item['sha256'], []).append(rel)
    receipts, all_objects, findings = [], set(), []
    with tempfile.TemporaryDirectory(prefix='source-verification-', dir=HERE) as temp:
        gitdir = Path(temp) / 'verification.git'
        env = dict(os.environ)
        for key in list(env):
            if key.startswith('GIT_'):
                del env[key]
        env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null', GIT_TERMINAL_PROMPT='0')
        subprocess.run(['git', 'init', '--bare', '--quiet', str(gitdir)], check=True, env=env)
        (gitdir/'objects/info/alternates').write_text(str(objects) + '\n')
        def git(*args, data=None):
            return subprocess.run(['git', '--git-dir='+str(gitdir), *args], input=data,
                                  check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env).stdout
        for sha, paths in sorted(unique.items()):
            value = (source/paths[0]).read_bytes()
            assert digest(value) == sha
            git('bundle', 'verify', str(source/paths[0]))
            pack = value.split(b'\n\n', 1)[1]
            result = git('index-pack', '--stdin', '--fix-thin', '--strict', data=pack).decode().strip()
            pack_hash = result.split()[-1]
            idx = gitdir/'objects/pack'/('pack-'+pack_hash+'.idx')
            rows = git('verify-pack', '-v', str(idx)).decode().splitlines()
            object_ids = [r.split()[0] for r in rows if len(r.split()[0])==40]
            types, text_objects, text_bytes = {}, 0, 0
            for oid in object_ids:
                typ = git('cat-file', '-t', oid).decode().strip()
                types[typ] = types.get(typ,0)+1
                # Pack objects only, including resolved delta bases; do not traverse
                # unrelated objects from the read-only alternate source repository.
                data = git('cat-file', typ, oid)
                is_text, hits = scan(data, 'bundle:'+sha+':object:'+oid)
                text_objects += is_text
                text_bytes += len(data) if is_text else 0
                findings.extend(hits)
                all_objects.add(oid)
            receipts.append(dict(sha256=sha, paths=paths, bytes=len(value),
                                 header=bundle_header(value), prerequisite_verification='pass',
                                 strict_pack_verification='pass', packed_objects=len(object_ids),
                                 object_types=types, text_objects_scanned=text_objects, text_bytes_scanned=text_bytes))
        final = json.loads((source/'final-source.json').read_text())
        checked = {}
        for path, sha in final['files'].items():
            data = git('show', final['head']+':'+path)
            assert digest(data) == sha, path
            checked[path] = sha
        patch = git('diff', '--abbrev=10', final['baseline'], final['head'], '--')
        assert digest(patch) == final['diff']['sha256'] == digest((source/'final-source.diff').read_bytes())
        git('read-tree', final['baseline'])
        git('apply', '--cached', data=patch)
        restored_tree = git('write-tree').decode().strip()
        assert restored_tree == git('rev-parse', final['head']+'^{tree}').decode().strip()
    result = dict(state='pass' if not findings else 'requires_review', unique_bundles=len(unique),
        bundle_paths=sum(len(p) for p in unique.values()), unique_git_objects_inspected=len(all_objects),
        bundles=receipts, credential_scan_findings=findings,
        final_source=dict(baseline=final['baseline'], head=final['head'],
                          all_six_blob_sha256_verified=checked, exact_original_diff_sha256=digest(patch),
                          patch_index_abbreviation=10, patch_reconstructed_tree=restored_tree),
        isolation='Only disposable bare Git object storage under archive-prep was written. Existing repository objects were read through an alternate. Temporary verification storage was removed.',
        science_executed=False, remote_access=False)
    (HERE/'bundle-verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('bundles','final_source')}, indent=2))
    if findings:
        raise ValueError('Bundle text scan needs review')


if __name__ == '__main__':
    main()
