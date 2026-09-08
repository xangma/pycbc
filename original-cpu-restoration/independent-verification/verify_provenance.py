"""Independent local Git, archive and Linux test receipt verification."""
import ast
import json
from pathlib import Path
import subprocess
import tarfile
import xml.etree.ElementTree as ET

from verify_evidence import O, OUT, ORIGINAL, V2, git, read, require, sha, sha_data, snapshot


def setup_ast(head):
    tree = ast.parse(git('show', head + ':setup.py'))
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if isinstance(body, list):
            node.body = [n for n in body if not (isinstance(n, ast.Expr) and
                         isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'extras_require' for t in node.targets):
            require(isinstance(node.value, ast.Dict), 'Unexpected extras syntax')
            pairs = [(k, v) for k, v in zip(node.value.keys, node.value.values)
                     if not (isinstance(k, ast.Constant) and k.value == 'torch')]
            node.value.keys = [k for k, _ in pairs]
            node.value.values = [v for _, v in pairs]
    return ast.dump(tree, include_attributes=False)


def ancestor(a, b):
    result = subprocess.run(['git', '-C', str(O.parents[2]), 'merge-base', '--is-ancestor', a, b])
    require(result.returncode in (0, 1), 'Git ancestry error')
    return result.returncode == 0


def main():
    manifest, recorded = read(O / 'manifest-v2.json'), read(O / 'source-preservation.json')
    require(manifest['cpu_base'] == recorded['original_cpu'] == ORIGINAL and recorded['tested_main'] == V2, 'Source proof pin mismatch')
    native_paths = ['*.pyx', '*.pxd', '*.pxi', '*.c', '*.cc', '*.cpp', '*.h', '*.hpp', '*.cu', '*.cuh', 'pycbc/lib']
    prs = []
    baseline_setup = setup_ast(ORIGINAL)
    for row in manifest['prs']:
        head, number = row['new_head'], str(row['pr'])
        require(ancestor(ORIGINAL, head) and ancestor(row['new_base'], head) and
                not ancestor('66789ac4a7468094b0cc3ca1498a1de67e0311f6', head), 'Ancestry mismatch')
        original_diff = git('diff', '--name-only', ORIGINAL, head, '--', *native_paths).decode().splitlines()
        restoration_diff = git('diff', '--name-only', row['old_head'], head, '--', *native_paths).decode().splitlines()
        require(not restoration_diff and (not original_diff or number == '17'), 'Unexpected native source change')
        require(recorded['native_unchanged'][number] == dict(restoration=True, versus_original=not bool(original_diff),
                preexisting_optional_native_files=original_diff), 'Recorded native proof mismatch')
        require(setup_ast(head) == baseline_setup and recorded['setup_ast_original_except_optional_torch'][number], 'Setup AST mismatch')
        prs.append(dict(pr=row['pr'], head=head, original_ancestor=True, corrected_cpu_commit_absent=True,
                        declared_base_ancestor=True, native_restoration_unchanged=True,
                        optional_native_differences=original_diff, setup_ast_equal_except_torch_extra=True))
    require(len(prs) == 15 and {r['pr'] for r in prs} == {r['pr'] for r in recorded['prs']}, 'PR field sets mismatch')
    fft = {}
    for path in ('pycbc/fft/mkl.py', 'pycbc/fft/fftw.py', 'pycbc/fft/npfft.py'):
        a, b = git('show', ORIGINAL + ':' + path), git('show', V2 + ':' + path)
        require(a == b and sha_data(a) == recorded['byte_exact_original_files'][path], 'CPU FFT source differs')
        fft[path] = sha_data(a)
    archive = O / 'campaign-v2-results.tar.gz'
    expected = 'c93f88b0a34d5ef89b533867f8c997553257fe56d0dda443a401e3f120d59264'
    require(sha(archive) == expected, 'Downloaded tar digest mismatch')
    bundle = O / 'campaign-v2-results'
    local = snapshot(bundle)
    archived_files = {}
    with tarfile.open(archive) as stream:
        for member in stream:
            if member.isfile():
                name = member.name.removeprefix('./')
                require(name not in archived_files, 'Duplicate tar member')
                archived_files[name] = sha_data(stream.extractfile(member).read())
    require(local == archived_files, 'Extracted bundle file sets or bytes differ from tar')
    require(sha(archive) == expected, 'Tar changed during verification')
    status = read(bundle / 'status.json')
    require(status['state'] == 'complete' and set(status['completed']) ==
            {'qual-' + a for a in ('original-cpu', 'proposed-cpu', 'torch-cpu', 'torch-cuda')}, 'Incomplete campaign')
    staged = O / 'campaign-v2'
    pins = read(bundle / 'harness-pins.json')
    require(all(sha(staged / name) == value for name, value in pins.items()
                if (staged / name).exists()), 'Downloaded/staged harness mismatch')
    extra_helpers = sorted(p.name for p in bundle.glob('*.py') if p.name not in pins)
    require(extra_helpers == ['linux-checks.py'] and sha(bundle / extra_helpers[0]) == sha(O / extra_helpers[0]),
            'Unaccounted bundled helper')
    linux = O / 'linux-checks'
    receipt = read(linux / 'status.json')
    require(receipt['head'] == V2 and receipt['state'] == 'complete', 'Linux receipt pin mismatch')
    runtime, = [r for r in receipt['results'] if r['name'] == 'runtime']
    require(runtime['returncode'] == 0, 'Linux runtime tests failed')
    cases = list(ET.parse(linux / 'runtime.xml').iter('testcase'))
    skipped = sum(c.find('skipped') is not None for c in cases)
    failed = sum(c.find('failure') is not None or c.find('error') is not None for c in cases)
    require((len(cases), skipped, failed) == (138, 1, 0), 'Linux XML counts differ')
    result = dict(status='pass', verifier_sha256=sha(__file__), original=ORIGINAL, main=V2,
                  source_preservation_receipt_sha256=sha(O / 'source-preservation.json'),
                  source_proof_recomputed=True, prs=prs, cpu_fft_file_sha256=fft,
                  tar_sha256=expected, extracted_file_set_and_hashes_match_tar=True, extracted_files=len(local),
                  all_four_runs_complete=True, staged_harness_matches_download=True,
                  extra_unpinned_bundle_helpers=extra_helpers,
                  extra_helper_scope='linux-checks.py is a separate test helper, not part of the pinned campaign harness.',
                  linux_runtime=dict(passed=137, skipped=1, failed=0, receipt_sha256=sha(linux / 'status.json'),
                                     junit_sha256=sha(linux / 'runtime.xml'), command=runtime['command']),
                  limitation='Native .so and external input bytes absent locally; their remote hashes remain primary-agent evidence.',
                  performance_claim=False)
    (OUT / 'provenance-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    (OUT / 'provenance-verification.md').write_text(
        '# Independent provenance verification\n\nPASS: all 15 pinned ancestry/native/setup checks reproduce the source-preservation receipt. '
        'Original 40e94792 is an ancestor and the CPU-correction commit 66789ac4 is absent. '
        'Main aa6b795a has byte-identical original MKL, FFTW and NumPy FFT files. '
        'PR17 retains only the acknowledged pre-existing optional native differences; the restoration changes no native source versus the published heads.\n\n'
        'The downloaded tar SHA-256 matches the supplied remote digest, and every extracted regular-file name and digest matches the tar. '
        'All four campaign receipts are complete. Staged and downloaded campaign harness hashes match. '
        'The bundle additionally contains linux-checks.py, an unpinned separate test helper, matching its local source.\n\n'
        'Linux runtime JUnit evidence independently counts 137 passed, 1 skipped, 0 failures at aa6b795a. '
        'This is verification of recorded results, not a fresh Linux test run.\n\n'
        'Linux .so binaries and external bank/frame bytes are absent locally; remote rehash verification remains primary-agent evidence. '
        'No performance claim is made.\n')
    print(json.dumps({k: result[k] for k in ('status', 'source_proof_recomputed', 'tar_sha256', 'extracted_files', 'all_four_runs_complete')}))


if __name__ == '__main__':
    main()
