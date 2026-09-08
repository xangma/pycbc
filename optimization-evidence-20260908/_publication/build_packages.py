"""Deterministically package the five input snapshots; write only under archive-prep."""
import collections
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

from inspect_inputs import HERE, scan

PACKAGES = HERE / 'packages'
SPEC = {
    'torch-fft-optimization-20260908': {
        'outcome': 'Promoted CPU workspace and common frame-loader improvements qualified; CPU parity unmet. CUDA scalar scheduling showed a bounded warm-API benefit, without demonstrated executable gain in its separate campaign.',
        'main_report': 'HANDOFF.md',
        'primary_summaries': ['loader-v1-summary.json', 'loader-v1-metrics.csv', 'executable-v3-summary.json', 'cpu-timing-v5-summary.json'],
        'scope': 'All successful and rejected CPU alternatives, superseded executable v1/v2 harness failures, executable v3, loader qualification/timing, warm CUDA diagnostics, native and local gates, peer reviews, source bundles and exact patches.'},
    'torch-residual-optimization-20260908': {
        'outcome': 'Descriptor reuse passed qualification and all four paired timing repeats; median wall reductions were 2.213184% for standard CPU and 7.665707% for CUDA. The separate NumPy-copy CPU candidate was rejected.',
        'main_report': 'HANDOFF.md',
        'primary_summaries': ['descriptor-reuse-timing-v1-summary.json', 'descriptor-reuse-timing-v1-samples.csv', 'descriptor-reuse-v1-summary.json', 'cpu-copy-v1-summary.json', 'cpu-stage-v1-summary.json'],
        'scope': 'Complete CPU stage/copy diagnostics, descriptor diagnostic and timing inputs/results, native and science gates, all raw HDF/timing outputs, superseded unlaunched freeze and independent reviews.'},
    'torch-offline-cuda-graph-20260908': {
        'outcome': 'Version 2 passed the frozen rule: median complete-child wall 20.257390172453597 to 19.94459447549889 seconds, a 1.5441065916776076% reduction, with all four pairs favorable. Version 1 failed qualification and ran no timing samples.',
        'main_report': 'OFFLINE-CUDA-GRAPH-v2.md',
        'primary_summaries': ['acquired-diagnostic-v2/timing-summary.json', 'offline-cuda-graph-v2-samples.csv', 'audit-v2.json', 'peer-results-review-v2.json'],
        'scope': 'Both fixed diagnostic versions, original input/results archives, frozen native/executable gates, raw HDF/science comparisons, eight timing samples and closure evidence.'},
    'torch-cpu-workspace-policy-20260908': {
        'outcome': 'Rejected before timing: DFTI_AVOID changed retained complex128 native bytes although final complex64 bytes and both FFTW limits passed. Zero timing workers ran; the frozen native-and-final byte-parity rule was not relaxed.',
        'main_report': 'CPU-WORKSPACE-POLICY-v1.md',
        'primary_summaries': ['acquired-diagnostic-v1/qualification.json', 'owner-results-audit.json', 'peer-results-review-v1.json'],
        'scope': 'All frozen input/result files, ABI/runtime pins, native contracts, first-case qualification failure, controller stop decision and terminal closure.'},
    'torch-profiling-investigation-20260908': {
        'outcome': 'Review and diagnostic synthesis, including failed precision/copy/workspace/page-backing alternatives and accepted descriptor reuse. Historical reviews are preserved with their original scope and language.',
        'main_report': 'RESIDUAL-ANALYSIS-v4.md',
        'primary_summaries': ['residual-analysis-v4.json', 'residual-analysis-v3.json', 'page-backing-summary-v2.json', 'page-backing-results-v2-review.json', 'sync-attribution.json', 'existing-cpu-summary.json'],
        'scope': 'Every non-cache file in the requested investigation root. This root has no preexisting outer SHA256SUMS; the publication creates a fresh inventory. Page-backing reviews and summaries are included; the separate page-backing campaign root and its raw archives were not among the five inputs and are external dependencies.'},
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def meta(data):
    return dict(bytes=len(data), sha256=sha(data))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def write_json(path, obj):
    write(path, (json.dumps(obj, indent=2, sort_keys=True) + '\n').encode())


def compressed(data):
    result = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=result, mtime=0, compresslevel=9) as stream:
        stream.write(data)
    return result.getvalue()


def make_tar(entries):
    raw = io.BytesIO()
    rows = []
    with tarfile.open(fileobj=raw, mode='w', format=tarfile.USTAR_FORMAT) as archive:
        for name, data in sorted(entries.items()):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = 0o644
            member.mtime = member.uid = member.gid = 0
            member.uname = member.gname = ''
            archive.addfile(member, io.BytesIO(data))
            rows.append(dict(name=name, type='file', bytes=len(data), mode=0o644, uid=0, gid=0,
                             mtime=0, pax_headers={}, sha256=sha(data)))
    return raw.getvalue(), rows


def read_source(root, name, item):
    data = (root / name).read_bytes()
    if meta(data) != {k: item[k] for k in ('bytes', 'sha256')}:
        raise ValueError('Source changed after inspection: ' + str(root / name))
    return data


def bundle_header(data):
    lines = data.split(b'\n\n', 1)[0].decode().splitlines()
    return dict(header=lines[0], prerequisites=[s[1:] for s in lines[1:] if s.startswith('-')],
                refs=[s for s in lines[1:] if not s.startswith(('-', '@'))],
                capabilities=[s for s in lines[1:] if s.startswith('@')])


def main():
    inspection = json.loads((HERE / 'input-inspection.json').read_text())
    if inspection['scan']['findings']:
        raise ValueError('Unresolved text scan findings')
    PACKAGES.mkdir()
    documentation, summaries = {}, []
    for name, spec in SPEC.items():
        incoming = inspection['roots'][name]
        root = Path(incoming['source_root'])
        out = PACKAGES / name
        out.mkdir()
        originals, storages, representatives, supplemental = incoming['files'], {}, {}, {}
        transports, archives, readable = {}, {}, []
        top_views = [p for p in originals if '/' not in p and
                     (p.endswith(('.md', '.csv')) or (p.endswith('.json') and
                      any(s in p for s in ('summary', 'analysis', 'audit', 'source', 'review', 'seal'))))]
        source_views = [p for p in originals if p.endswith(('.bundle', '.diff', '.patch'))]
        selected_views = sorted(set(top_views + source_views + spec['primary_summaries']))
        for rel in selected_views:
            value = read_source(root, rel, originals[rel])
            dest = 'readable/' + rel
            write(out / dest, value)
            storage = dict(kind='readable', path=dest)
            storages[rel] = storage
            representatives.setdefault(originals[rel]['sha256'], storage)
            readable.append(dict(original_path=rel, path=dest, **originals[rel]))
        for rel, record in sorted(incoming['archives'].items()):
            value = read_source(root, rel, originals[rel])
            path = 'transport/' + rel + '.gz'
            packed = compressed(value)
            write(out / path, packed)
            transports[rel] = dict(path=path, packed=meta(packed), original=originals[rel])
            storages[rel] = dict(kind='gzip', path=path)
            archives[rel] = record
            for member in record['members']:
                if member['type'] != 'file':
                    continue
                member_name = member['name'].removeprefix('./')
                representatives.setdefault(member['sha256'], dict(kind='archive', archive=rel, member=member_name))
        for rel, record in sorted(originals.items()):
            if rel in storages:
                continue
            # Rehash every selected file even if transported via an exact-byte alias.
            value = read_source(root, rel, record)
            if record['sha256'] in representatives:
                storages[rel] = representatives[record['sha256']]
            else:
                member = 'blobs/' + record['sha256']
                supplemental[member] = value
                storage = dict(kind='supplement', member=member)
                storages[rel] = storage
                representatives[record['sha256']] = storage
        raw, members = make_tar(supplemental)
        packed = compressed(raw)
        write(out / 'supplement.tar.gz', packed)
        archives['supplement'] = dict(**meta(raw), members=members)
        coverage = dict(selected_files=len(originals), selected_bytes=sum(r['bytes'] for r in originals.values()),
                        original_seal=incoming['original_seal'], interpreter_cache_omissions=incoming['omitted'],
                        omitted_evidence_files=0,
                        storage_counts=dict(collections.Counter(v['kind'] for v in storages.values())),
                        original_archives=len(transports),
                        original_archive_regular_members=sum(m['type']=='file' for r in incoming['archives'].values() for m in r['members']),
                        supplement_unique_files=len(supplemental),
                        preserved_source_files=[p for p in originals if p.endswith(('.py', '.bundle', '.diff', '.patch'))])
        manifest = dict(schema='pycbc-publication-evidence-v1', root_name=name, source_root=str(root),
                        original_seal=incoming['original_seal'],
                        files={p: dict(**r, storage=storages[p]) for p,r in sorted(originals.items())},
                        transports=transports, supplement=dict(path='supplement.tar.gz', packed=meta(packed), original=meta(raw)))
        write_json(out / 'manifest.json', manifest)
        write_json(out / 'archive-inventory.json', archives)
        write_json(out / 'coverage.json', coverage)
        write_json(out / 'readable-sources.json', readable)
        inventories = incoming['authoritative_inventories']
        frozen = [m for m in inventories if Path(m['path']).name in ('manifest.json','results-manifest.json')]
        missing_frozen = [dict(manifest=m['path'], **e) for m in frozen for e in m['entries'] if e['status']=='external_or_historical']
        if missing_frozen:
            raise ValueError('Frozen input/results entry missing from package')
        write_json(out / 'evidence-inventories.json', dict(
            inventory_scope='Top-level SHA256 entries in manifest, pins and source JSON records; nested records remain preserved unchanged.',
            frozen_manifest_count=len(frozen), frozen_entry_count=sum(len(m['entries']) for m in frozen),
            all_frozen_input_result_entries_preserved=True, inventories=inventories))
        source_records = []
        for rel in originals:
            if rel.endswith(('.bundle', '.diff', '.patch')) or any(s in Path(rel).name for s in ('source-pins', 'source-staging', 'source.json', 'final-source')):
                row = dict(path=rel, **originals[rel])
                if rel.endswith('.bundle'):
                    row['bundle'] = bundle_header((root / rel).read_bytes())
                source_records.append(row)
        write_json(out / 'source-inventory.json', source_records)
        write_json(out / 'publication-summary.json', dict(**spec, **{k:v for k,v in coverage.items() if k not in ('interpreter_cache_omissions','preserved_source_files')},
            result_interpretation='Copied from original reports; no new scientific computation or performance measurement.',
            caveats=['Fixed finite workloads on a shared host; reported ranges are observed ranges, not confidence intervals.',
                     'Full-executable, isolated API, native, setup and qualification timings retain their separate meanings.',
                     'Historical paths, runtime identities and process closure records are provenance, not current observations.',
                     'Raw input bank/GWF, full source baselines, native binaries and environments are external pinned dependencies.']))
        write_json(out / 'text-scan.json', dict(state='pass', candidate_findings=0,
            scope='UTF-8 text without NUL in every selected file and every regular member of every preserved original tar; no content rewritten.',
            allowed_provenance='Public gravitational-wave inputs, environment paths/hostnames, source identities and experiment process records.',
            review='Broad credential/private-data keyword scan yielded five uses of the word authorization in experiment review prose, all unrelated to credentials. No unrelated private data identified.',
            binary_limit='Binary scientific outputs are preserved by hash; Git bundle object text is addressed separately in the preparation verification report.'))
        write(out / 'verify.py', (HERE / 'verify_package.py').read_bytes())
        main_report = 'readable/' + spec['main_report']
        report = f'''# {name}: publication evidence

{spec['outcome']}

Read [{spec['main_report']}]({main_report}) for the original report and limitations.
`publication-summary.json` is a navigation aid; `readable-sources.json` maps every
readable copy to its exact original path, SHA256, length and mode.

{spec['scope']}

This package preserves **{len(originals)} of {len(originals)} selected evidence files**
({coverage['selected_bytes']:,} original bytes) and all {len(transports)} original archives.
Only {len(incoming['omitted'])} interpreter-cache files are excluded, with individual hashes
in `coverage.json`. Original archives are transported losslessly as timestamp-zero,
level-9 gzip streams; original tar bytes, names, metadata and checksums reconstruct
unchanged. Files already present in those archives use exact-byte member references.
Other duplicate files share a content-addressed supplemental blob. No redundant
restored tree is shipped. Small readable report/summary/source copies are deliberate.

`manifest.json` inventories every logical file and its storage mapping.
`archive-inventory.json` inventories every original tar member plus the deterministic
supplement; `source-inventory.json` identifies exact patches, bundles and source pins.
`evidence-inventories.json` preserves every discovered flat source/input/run hash
inventory and records matching payloads or external/historical dependencies.
All {sum(len(m['entries']) for m in frozen)} entries of {len(frozen)} frozen input/result
manifests resolve to preserved bytes. The original outer checksum seal is preserved
when present; absence is explicit in `coverage.json`.

## Verify and reconstruct

Use Python 3.10+ and independently supplied publisher hashes for authenticity:

```sh
python3 -I verify.py
python3 -I verify.py --restore /absolute/fresh/parent/{name}
```

Verification checks the full physical package, decompresses and checks original
archive bytes and all members, resolves and hashes every logical file, and checks
the original outer seal. Restoration requires an unused destination whose parent
exists without symlink aliases, outside this package. It validates all payloads
before writing and rehashes every exclusively written file. Original selected file
modes are restored; tar ownership and timestamps are not applied. The helper rejects
traversal, links, duplicate members, missing/extra files and file/directory conflicts.
It never imports or executes experiment code, contacts a remote system or changes
the source evidence. `SHA256SUMS` covers every physical file except itself.

## Regenerate plots from recorded evidence

Primary machine files are listed below; they contain recorded raw samples and/or
qualified decisions. All accompanying raw worker logs, timing JSON/CSV, HDF/PSD
outputs, qualification receipts, comparators, frozen protocols and existing
summarizers present in the source inventory reconstruct at their original relative
paths. Read those inputs to generate plots in a new directory. Preserve distinctions
between qualification, warm API, native, internal and complete-child measurements;
do not pool superseded or failed campaigns with the accepted timing samples.
Existing scripts retain historical absolute paths. Review and relocate their input
and output constants in disposable working copies before any optional replay.
This preparation ran no acquisition, FFT, runtime tests, plot generator or scientific
comparator. Integrity verification does not re-observe historical scientific gates.

''' + '\n'.join(f'- [`{p}`](readable/{p})' for p in spec['primary_summaries']) + '''

The public input bank and GWF are identified by their original SHA256 pins, but
their bytes are not in these five input roots. Full source baselines, native
libraries, environments and predecessor campaign roots are likewise external.
Incremental Git bundles require the exact prerequisite commits listed in
`source-inventory.json`; they are not complete checkouts. The FFT package preserves
the final four-commit bundle/diff from 7e56ac42417dd6f61498f917e04c5a55250ddc26 to
ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f. Other packages refer to that pinned source
and preserve their own experiment helpers. A new experimental rerun requires those
dependencies and a separately reviewed execution environment.

Original acquisition-time statements such as “no publication” remain verbatim;
this package is a later publication preparation, not a rewrite of the acquisition.
No source integration, PR change, remote mutation or publication is performed here.
'''
        write(out / 'README.md', report.encode())
        physical = sorted(p for p in out.rglob('*') if p.is_file())
        findings = []
        for path in physical:
            is_text, hits = scan(path.read_bytes(), path.relative_to(out).as_posix())
            findings.extend(hits)
        if findings:
            raise ValueError('Outgoing text scan requires review: ' + repr(findings))
        write(out / 'SHA256SUMS', ''.join(f'{sha(p.read_bytes())}  {p.relative_to(out).as_posix()}\n' for p in physical).encode())
        size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        documentation[name] = dict(main_report=f'packages/{name}/{main_report}',
            machine_summaries=[f'packages/{name}/readable/{p}' for p in spec['primary_summaries']],
            all_readable_sources=f'packages/{name}/readable-sources.json',
            logical_inventory=f'packages/{name}/manifest.json', scope=spec['scope'])
        summaries.append(dict(name=name, path=str(out), physical_bytes=size,
            physical_files=len(physical)+1, selected_files=len(originals), selected_bytes=coverage['selected_bytes'],
            original_archives=len(transports), original_archive_regular_members=coverage['original_archive_regular_members'],
            frozen_input_result_entries=sum(len(m['entries']) for m in frozen),
            original_seal_entries=incoming['original_seal']['entries'],
            checksum_file_sha256=sha((out/'SHA256SUMS').read_bytes())))
        print(json.dumps(summaries[-1]), flush=True)
    write_json(HERE / 'documentation-index.json', documentation)
    write_json(HERE / 'package-summary.json', summaries)


if __name__ == '__main__':
    main()
