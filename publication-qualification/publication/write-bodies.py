#!/usr/bin/env python3
"""Generate seven local PR bodies from frozen sources and reviewed evidence.

No Git, network, PR, label or source-worktree mutations. This does not publish.
The required context supplies reviewed claims; file hashes and JSON assertions
must still match when generation runs. Existing output is never overwritten.
"""

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile


HERE = Path(__file__).resolve().parent
NUMBERS = (8, 9, 11, 15, 19, 16, 17)
UNCHANGED = {8, 9, 11}
OLD_HEADS = {
    8: '0d6160c3ed22d87e63de4a0c663ac61038799cf4',
    9: '9db3c9779c45b71d130f6f56e9be84a385780b86',
    11: 'a4ba198e7c4d2fa1b0238de946f1367a98e7a2f6',
    15: 'dfd42bf76766cadca0eecf609a1eaeac73534676',
    19: 'fa38dbca79f4e2e662f079e56b4d9dbb73fd9ef4',
    16: '37d3c6b4ac1ec74a2dcd42558f44f21b92218ca5',
    17: 'bd53914be6d2e4324cc867d52b3842b77cc6729a',
}
PARENT_15 = '78d99b5e0f540abd438e01a221de77fc02109b3f'
FINAL_SOURCE = '837f38d493420043e45fb1ad210a0ccf68bacbaa'
OPTIMIZED_SOURCE = 'a4d77a6d1863c0515e8dace64c5609b63d40b51e'
HEADINGS = (
    'Standard information about the request', 'Motivation', 'Contents',
    'Links to any issues or associated PRs', 'Testing performed',
    'Additional notes',
)
COC = ('- [ ] The author of this pull request confirms they will adhere to the '
       '[code of conduct](https://github.com/gwastro/pycbc/blob/master/'
       'CODE_OF_CONDUCT.md)')
NOTE = ('*AI Agent Note: Unchecked by default. @xangma, please review this PR '
        'and check the Code of Conduct box above to confirm your agreement '
        'before requesting review.*')
CREATOR = 'This PR was created by AI Gareth'
IMMUTABLE = re.compile(
    r'https://github\.com/xangma/pycbc/(?:blob|tree)/[0-9a-f]{40}(?:/[^\s)]*)?$')
DOWNSTREAM = {
    8: 'Later batch-overlap admission and spectral-power precision improvements '
       'are in downstream {pr15}. '
       'This filtering head is unchanged; the new timing evidence does not '
       'measure or establish performance on this head.',
    9: 'Later CPU peak-reduction and accelerator fast-path improvements are '
       'in downstream {pr15}. This search head is unchanged; the new timing '
       'evidence does not measure or establish performance on this head.',
    11: 'Later sparse float64 phase-polynomial evaluation is in downstream '
        '{pr15}; that change retains the original float32 path. This waveform '
        'head is unchanged; the new timing evidence does not measure or '
        'establish performance on this head.',
    19: 'This replacement head retains the formatting changes, marks the existing '
        'FFT public re-exports explicitly, and suppresses the redundant exception '
        'context for the existing optional cuFFT import error. It inherits '
        'the performance, executable-context and precision fixes from {pr15}. '
        'Those fixes are outside this PR’s own diff.',
    16: 'The optional FFT features are preserved; public-export and import-error '
        'lint cleanup is supplied by the formatting base. This replacement head '
        'inherits the final fixes from {pr15}; qualification below identifies '
        'the sources used for historical timing and fresh final-commit tests.',
    17: 'The optional CPU peak/chi-square kernels and native admission gate '
        'are preserved. CPU FFT tuning retains direct plans from 32768 through '
        '524288 samples. The 1048576, 2097152 and 4194304 sample inverse FFTs '
        'use the qualified double-precision workspace only with one thread; '
        'other thread counts retain the fallback. New dispatch tests cover '
        'this integration. Native peak dispatch remains ahead of '
        'the improved Torch fallback inherited from {pr15}. The later '
        'accelerator peak refinement leaves the CPU reduction algorithm '
        'unchanged. Inherited precision and executable-context fixes remain '
        'outside this PR’s own feature diff.',
}
SUMMARY_15 = (
    'Improve Torch batch performance, repair executable-context and numerical '
    'precision handling, and validate the compressed BNS/NSBH pycbc_inspiral '
    'workload against a tuned, single-thread normal MKL reference. Optimize '
    'compressed-waveform interpolation and qualified large search inverse FFTs.')
CONTENTS_15 = (
    'Replace pairwise output-overlap checks with a sorted span scan, bound CPU '
    'peak-reduction temporaries by chunking, and skip absent logarithmic terms '
    'in float64 TaylorF2 phase evaluation. Restore the direct accelerator '
    'peak-reduction path while preserving the CPU algorithm and the original '
    'float32 waveform path.\n\n'
    'Report executable CPU-thread counts safely and preserve the shared '
    'Torch processing context when deep-copying arrays. Correct spectral-power '
    'accumulation, PSD/strain precision handling and chi-square precision; '
    'correct the cumulative-power test reference and its rounding bound; '
    'and preserve the CUDA phase coefficient at the Python launch site with '
    'the kernel body unchanged. Reuse the compiled CPU linear interpolation '
    'routine through shared array views, and route qualified single-thread '
    'large inverse FFTs through a retained double-precision MKL workspace. '
    'These changes comprise eight fix commits '
    f'through `{OPTIMIZED_SOURCE}`. Regression tests cover the affected numerical '
    'and context behavior alongside overlap, peak, waveform and differentiation '
    'checks.\n\n'
    'The primary evidence uses compressed BNS/NSBH templates in pycbc_inspiral: '
    'retain the measured single-thread normal CPU/MKL tuning choice after '
    'verifying its implementation is unchanged, then run fresh normal CPU, '
    'Torch CPU and Torch CUDA comparisons at the selected geometry. The '
    f'earlier source `{FINAL_SOURCE}` supplies the separate before comparison. '
    'The supplement includes independent waveform and boundary validation, '
    'unprofiled timing repetitions and separate Python, native CPU and CUDA '
    'profiles. Testing performed records the reviewed results and accounting '
    'limits.\n\n'
    'Also retains disabled-chi-square fallback handling, benchmark harnesses, '
    'artifact validation, CPU/GPU test selectors and the validation/performance '
    'guides. Earlier paired v1 component measurements and the separate v2 '
    'accelerator follow-on remain supporting evidence at their actual measured '
    'revisions, including reused baseline/control attribution. Earlier '
    'executable attempts remain supplemental and are not reused as final runs. '
    'Historical FFT, inference and Triton evidence keeps its original source '
    'identity; optional TaylorF2 Triton support is inherited from #11.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(value):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{40}', value),
            'Expected full source SHA: ' + str(value))
    return value


def read_json(path, inputs):
    path = path.resolve()
    raw = path.read_bytes()
    inputs[str(path)] = digest(raw)
    return json.loads(raw)


def split_body(body):
    matches = list(re.finditer(r'^## (.+)$', body, re.M))
    require([m.group(1) for m in matches] == list(HEADINGS),
            'Unexpected snapshot template sections')
    sections = {m.group(1): body[m.end():matches[i + 1].start()
                if i + 1 < len(matches) else len(body)].strip()
                for i, m in enumerate(matches)}
    return body[:matches[0].start()].strip(), sections


def json_pointer(value, pointer):
    require(isinstance(pointer, str) and (pointer == '' or pointer.startswith('/')),
            'Invalid JSON pointer')
    for token in pointer.split('/')[1:] if pointer else []:
        key = token.replace('~1', '/').replace('~0', '~')
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def prose(text):
    require(isinstance(text, str) and 0 < len(text.strip()) <= 1800,
            'Missing or overlong reviewed prose')
    require(not re.search(r'^#|<!--|\[x\]|\[X\]|AI Agent Note|AI Gareth', text, re.M),
            'Reviewed prose cannot replace template or attribution')
    return text.strip()


def verify_context(context, path, inputs, snapshot_path, stack_path):
    require(context['schema'] == 'torch-performance-fix-body-context-v1'
            and context['reviewed'] is True, 'A reviewed context is required')
    require(context['snapshot_sha256'] == inputs[str(snapshot_path.resolve())],
            'Reviewed context refers to a different PR snapshot')
    require(context['candidate_stack_sha256'] == inputs[str(stack_path.resolve())],
            'Reviewed context refers to a different candidate stack')
    require(set(context['prs']) == {str(n) for n in NUMBERS},
            'Context must cover exactly seven PRs')
    evidence, documents = {}, {}
    for entry in context['evidence']:
        key = entry['id']
        require(isinstance(key, str) and key not in evidence, 'Duplicate evidence ID')
        require(IMMUTABLE.fullmatch(entry['url']), 'Evidence URL must pin a commit')
        item = (path.parent / entry['path']).resolve()
        raw = item.read_bytes()
        require(digest(raw) == entry['sha256'], 'Changed evidence: ' + str(item))
        inputs[str(item)] = digest(raw)
        require(entry['kind'] in {'report', 'tests', 'qualification', 'science',
                                  'artifact'},
                'Unknown evidence kind')
        assertions = entry.get('assertions', [])
        if entry['kind'] != 'artifact':
            require(assertions,
                    'Report/test/qualification/science evidence needs JSON assertions')
        if assertions:
            data = json.loads(raw)
            documents[key] = data
            for assertion in assertions:
                actual = json_pointer(data, assertion['pointer'])
                expected = assertion['equals']
                require(type(actual) is type(expected) and actual == expected,
                        f'Failed evidence assertion: {key} {assertion["pointer"]}')
        prose(entry['label'])
        evidence[key] = entry
    require({'report', 'tests', 'qualification', 'science'}
            <= {e['kind'] for e in evidence.values()},
            'Final report, test, qualification and science evidence are required')
    for head, identity in context['source_identities'].items():
        sha(head)
        sha(identity['tree'])
        prose(identity['label'])
    verify_primary_reference(context, evidence, documents)
    verify_optimized_reference(context, evidence, documents)
    return evidence


def verify_primary_reference(context, evidence, documents):
    primary = context['primary_reference']
    kinds = dict(source_manifest='qualification', provenance='qualification',
                 report='report', unit_tests='tests', waveform='science',
                 boundary='science')
    require(set(primary) == {'source_head', *kinds}
            and primary['source_head'] == FINAL_SOURCE
            and FINAL_SOURCE in context['source_identities'],
            'Primary reference must identify the final corrected source')
    ids = [primary[role] for role in kinds]
    require(len(ids) == len(set(ids)) and all(key in evidence for key in ids),
            'Primary reference requires six distinct evidence records')
    require(all(evidence[primary[role]]['kind'] == kind
                for role, kind in kinds.items()), 'Wrong primary evidence kind')
    records = {role: documents[primary[role]] for role in kinds}
    require(records['source_manifest']['commit'] == FINAL_SOURCE,
            'Primary source manifest identifies an earlier revision')
    proof = records['provenance']
    require(proof['status'] == 'pass' and proof['new_source_commit'] == FINAL_SOURCE
            and proof['normal_cpu_outputs_changed'] is True
            and proof['prior_science_reused'] is False,
            'Primary source provenance is incomplete or stale')
    report = records['report']
    require(report['schema_version'] == 1 and report['status'] == 'pass'
            and report['source_phases'] == {'precision': FINAL_SOURCE}
            and report['gates'] and all(v is True for v in report['gates'].values())
            and report['unit_tests']['source_commit'] == FINAL_SOURCE
            and report['parity']['status'] == 'pass'
            and report['source_provenance']['normal_cpu_outputs_changed'] is True
            and report['source_provenance']['prior_science_reused'] is False,
            'Primary report is incomplete, failed or from an earlier source')
    tests = records['unit_tests']
    require(tests['state'] == 'complete' and tests['passed'] is True
            and type(tests['returncode']) is int and tests['returncode'] == 0
            and tests['source_info'] == tests['source_after'] ==
            {'commit': FINAL_SOURCE, 'status': ''},
            'Final source unit-test receipt is incomplete or stale')
    for role, field, count in (('waveform', 'template_psd_pairs', 288),
                               ('boundary', 'cases', 36)):
        data = records[role]
        require(data['state'] == 'complete' and data['passed'] is True
                and data['completed_' + field] == data['expected_' + field] == count
                and data['source_before'] == data['source_after']
                and data['source_before']['commit'] == FINAL_SOURCE
                and not data['source_before']['status'],
                'Final scientific validation is incomplete or stale: ' + role)
    claims = context['prs']['15']['testing']
    cited = {key for claim in claims + context['prs']['15'].get('notes', [])
             for key in claim['evidence']}
    require(set(ids) <= cited, '#15 must cite all primary reference evidence')


def verify_optimized_reference(context, evidence, documents):
    optimized = context['optimized_reference']
    kinds = dict(source_manifest='qualification', report='report', unit_tests='tests',
                 ifft_matrix='qualification', ifft_decision='qualification', science='science',
                 waveform='science', boundary='science', compressed='science',
                 qualifications='qualification', measurements='qualification')
    require(set(optimized) == {'source_head', 'parent_source_head', *kinds}
            and optimized['source_head'] == OPTIMIZED_SOURCE
            and optimized['parent_source_head'] == FINAL_SOURCE
            and OPTIMIZED_SOURCE in context['source_identities'], 'Wrong optimized sources')
    ids = [optimized[role] for role in kinds]
    baseline_ids = {value for key, value in context['primary_reference'].items()
                    if key != 'source_head'}
    require(len(set(ids)) == 11 and not set(ids) & baseline_ids
            and all(key in evidence for key in ids), 'Eleven fresh optimized records are required')
    require(all(evidence[optimized[role]]['kind'] == kind for role, kind in kinds.items()),
            'Wrong optimized evidence kind')
    records = {role: documents[optimized[role]] for role in kinds}
    require(records['source_manifest']['commit'] == OPTIMIZED_SOURCE,
            'Optimized source manifest is stale')
    report = records['report']
    gates = {'source_provenance', 'unit_tests', 'large_ifft', 'campaign_complete',
             'trigger_parity', 'qualified_work_and_timings', 'waveform_validation',
             'boundary_validation', 'scientific_integration', 'profiles'}
    require(report['schema_version'] == 1 and report['status'] == 'pass'
            and report['source_commit'] == OPTIMIZED_SOURCE
            and report['parent_source_commit'] == FINAL_SOURCE
            and gates <= set(report['gates']) and all(v is True for v in report['gates'].values())
            and report['source_manifest_sha256'] == evidence[optimized['source_manifest']]['sha256']
            and report['parity']['status'] == 'pass' and len(report['parity']['comparisons']) == 18,
            'Optimized report is incomplete, failed or stale')
    for role in ('unit_tests', 'science', 'ifft_decision', 'compressed'):
        row = records[role]
        require(row['state'] == 'complete' and row['passed'] is True
                and row['source_info'] == row['source_after'] ==
                {'commit': OPTIMIZED_SOURCE, 'status': ''}, 'Failed optimized record: ' + role)
    require(type(records['unit_tests']['returncode']) is int
            and records['unit_tests']['returncode'] == 0, 'Optimized unit tests failed')
    matrix = records['ifft_matrix']
    require(matrix['state'] == 'complete' and matrix['source_before'] == matrix['source_after']
            and matrix['source_before']['head'] == OPTIMIZED_SOURCE
            and not matrix['source_before']['status'], 'Optimized IFFT matrix is incomplete or stale')
    for role, field, count in (('waveform', 'template_psd_pairs', 288), ('boundary', 'cases', 36)):
        row = records[role]
        require(row['state'] == 'complete' and row['passed'] is True
                and row['expected_' + field] == row['completed_' + field] == count
                and row['source_before'] == row['source_after']
                and row['source_before']['commit'] == OPTIMIZED_SOURCE
                and not row['source_before']['status'], 'Failed optimized science: ' + role)
    require(records['compressed']['expected_cases'] == records['compressed']['completed_cases'] == 576,
            'Incomplete compressed bank validation')
    for role, count in (('qualifications', 3), ('measurements', 16)):
        row = records[role]
        require(row['state'] == 'complete' and row['stage'] == role
                and type(row['returncode']) is int and row['returncode'] == 0
                and row['source_info'] == row['source_after'] ==
                {'commit': OPTIMIZED_SOURCE, 'status': ''}
                and len(row['plan']) == count and row['plan'] == row['completed'],
                'Incomplete optimized campaign: ' + role)
    claims = context['prs']['15']['testing']
    require(claims and optimized['report'] in claims[0]['evidence']
            and OPTIMIZED_SOURCE in claims[0]['source_heads'],
            '#15 must lead with the optimized report and its source')
    cited = {key for claim in claims + context['prs']['15'].get('notes', [])
             for key in claim['evidence']}
    require(set(ids) <= cited, '#15 must cite all optimized reference evidence')


def claim_text(claim, context, evidence):
    text = prose(claim['text'])
    heads = claim['source_heads']
    require(isinstance(heads, list) and heads and len(heads) == len(set(heads)),
            'Each claim needs distinct explicit source heads')
    require(all(h in context['source_identities'] for h in heads),
            'Claim has an unqualified source identity')
    refs = claim['evidence']
    require(isinstance(refs, list) and refs and len(refs) == len(set(refs))
            and all(key in evidence for key in refs), 'Claim evidence is missing')
    sources = '; '.join(f'{context["source_identities"][h]["label"]}: `{h}`'
                        for h in heads)
    links = ', '.join(f'[{evidence[k]["label"]}]({evidence[k]["url"]})' for k in refs)
    return f'{text}\n\nSources: {sources}. Evidence: {links}.'


def retained_limits(testing):
    return [p.strip() for p in re.split(r'\n\s*\n', testing) if re.search(
        r'(?i)(race remains unresolved|race remains|non-blocking CUDA peak-copy race|'
        r'Known CUDA limitation|nearly flat correlation peak|Comprehensive MPS)', p)]


def historical_links(body):
    links = []
    for label, url in re.findall(r'\[([^\]\n]+)\]\((https://[^)\s]+)\)', body):
        if IMMUTABLE.fullmatch(url) and url not in {u for _, u in links}:
            if label.lower() in {'new campaign', 'new evidence'}:
                label = 'archive'
            links.append((label, url))
    return ('Historical evidence retains its original source and timing attribution: '
            + '; '.join(f'[{label}]({url})' for label, url in links) + '.')


def render(pr, mapping, context, evidence):
    number = pr['number']
    summary, sections = split_body(pr['body'])
    testing = sections['Testing performed']
    require(testing.count('<!-- stack-index -->') == 1, 'Missing/duplicate stack index')
    prior_testing, stack = testing.split('<!-- stack-index -->')
    stack = '<!-- stack-index -->' + stack
    notes = sections['Additional notes']
    require(all(value in notes for value in (COC, NOTE, CREATOR, '`agent-assisted`')),
            'Missing required attribution/unchecked Code of Conduct note')
    require(not re.search(r'\[[xX]\].*code of conduct', notes, re.I), 'Checked CoC')
    new = mapping[number]
    links = sections['Links to any issues or associated PRs']
    expected = (f'Head branch: `{pr["headRefName"]}`. Head commit: '
                f'`{pr["headRefOid"]}`. Base branch: `{pr["baseRefName"]}`.')
    require(links.count(expected) == 1, 'Unexpected head/base description')
    links = links.replace(expected, expected.replace(pr['headRefOid'], new['head']))
    for other in mapping.values():
        links = links.replace('at `' + other['old_head'] + '`',
                              'at `' + other['head'] + '`')
    sections['Links to any issues or associated PRs'] = links
    pr15 = f'[#15]({mapping[15]["url"]}) at `{mapping[15]["head"]}`'
    if number == 15:
        summary = SUMMARY_15
        sections['Standard information about the request'] = (
            'This is a: efficiency update, bug fix, validation and documentation.\n\n'
            'This change affects: Torch filtering, TaylorF2 search/inference, '
            'processing contexts and pycbc_inspiral precision handling.\n\n'
            'This change changes: Python batch and precision implementation, '
            'executable context handling, regression tests, CI selectors, '
            'benchmark tooling and performance guides.\n\n'
            'This change: includes focused tests and source-attributed validation.\n\n'
            'This change will: retain Torch as an optional dependency.')
        sections['Motivation'] = (
            'Large batches exposed avoidable overlap-check, CPU temporary-array '
            'and phase-evaluation costs. Executable profiles identified costly '
            'compressed-template interpolation and large search inverse FFTs. '
            'Full executable validation also exposed '
            'context-copying and numerical precision failures. The fixes address '
            'both, with a tuned normal CPU/MKL reference, independent scientific '
            'checks and matched backend evidence on the final corrected source.')
        sections['Contents'] = CONTENTS_15
    else:
        if number == 19:
            summary = 'Format and lint the existing FFT modules before the optional FFT behavior changes.'
            sections[HEADINGS[0]] = sections[HEADINGS[0]].replace(
                'This is a: formatting and import-order cleanup.',
                'This is a: formatting, import-order and lint cleanup.').replace(
                'This change changes: source presentation and import ordering.',
                'This change changes: source presentation, import ordering, explicit public re-exports '
                'and the traceback context of an optional-import error.')
            old_scope = ('Batch execution, cache keys, wisdom handling, public-export aliases, '
                         'exception changes and new tests stay in the feature PR.')
            require(sections['Contents'].count(old_scope) == 1, 'Unexpected FFT formatting scope')
            sections['Contents'] = sections['Contents'].replace(old_scope,
                'Batch execution, cache keys, wisdom handling and new tests stay in the feature PR. '
                'The existing public-export aliases and optional-import exception context are '
                'made explicit here to satisfy the static checks.')
        sections['Contents'] += '\n\n' + DOWNSTREAM[number].format(pr15=pr15)
    if 'This change will:' not in sections[HEADINGS[0]]:
        sections[HEADINGS[0]] += (
            '\n\nThis change will: preserve the dependency scope described above.')
    reviewed = context['prs'][str(number)]
    require(1 <= len(reviewed['testing']) <= 4, 'Expected one to four testing claims')
    additions = [claim_text(c, context, evidence) for c in reviewed['testing']]
    extra = reviewed.get('notes', [])
    require(len(extra) <= 2, 'At most two additional evidence notes')
    additions += [claim_text(c, context, evidence) for c in extra]
    if number in UNCHANGED:
        additions.insert(0, f'Head `{new["head"]}` is unchanged. New performance '
                         f'measurements belong to downstream {pr15} and the '
                         'explicit source revisions below, not this PR head.')
    else:
        additions.insert(0, f'Publication head: `{new["head"]}`. Test and timing '
                         'records below identify their actual source revisions. '
                         'Verified patch preservation does not claim tests ran '
                         'on a later publication commit.')
    if number == 17:
        additions.append('The optional CPU before/after comparison qualifies the '
                         'recorded native-gate configuration and fallback; it '
                         'does not change the preserved native CPU kernels.')
    sections['Testing performed'] = '\n\n'.join(
        additions + retained_limits(prior_testing)
        + [historical_links(pr['body']), stack])
    result = summary + '\n\n' + '\n\n'.join(
        f'## {name}\n\n{sections[name]}' for name in HEADINGS) + '\n'
    require(stack in result and sections['Additional notes'] == notes,
            'Stack index or attribution changed')
    require(all(p in result for p in retained_limits(prior_testing)), 'Lost limitation')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=HERE / 'prs-current.json')
    parser.add_argument('--stack', type=Path, default=HERE / 'candidate-stack.json')
    parser.add_argument('--template', type=Path,
                        default=HERE.parents[2] / '.github/PULL_REQUEST_TEMPLATE.md')
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=HERE / 'bodies')
    args = parser.parse_args()
    require(not args.out.exists(), 'Output already exists; choose a new --out')
    require(args.out.parent.is_dir(), 'Output parent does not exist')
    inputs = {str(Path(__file__).resolve()): digest(Path(__file__).read_bytes())}
    prs = read_json(args.snapshot, inputs)
    stack = read_json(args.stack, inputs)
    context = read_json(args.context, inputs)
    template = args.template.read_bytes()
    inputs[str(args.template.resolve())] = digest(template)
    require(re.findall(r'^## (.+)$', template.decode(), re.M) == list(HEADINGS),
            'PR template changed; review generator sections')
    require(len(prs) == len(NUMBERS) and {p['number'] for p in prs} == set(NUMBERS),
            'Expected the seven frozen PRs')
    by_number = {p['number']: p for p in prs}
    for number, pr in by_number.items():
        require(pr['headRefOid'] == OLD_HEADS[number] and pr['state'] == 'OPEN'
                and isinstance(pr['isDraft'], bool)
                and 'agent-assisted' in {x['name'] for x in pr['labels']},
                f'Unexpected PR snapshot metadata: {number}')
    require(stack['schema'] == 'torch-performance-fix-candidates-v1', 'Stack schema')
    candidates = {c['number']: c for c in stack['candidates']}
    require(len(stack['candidates']) == 4
            and set(candidates) == set(NUMBERS) - UNCHANGED,
            'Expected four replacement candidates')
    mapping = {}
    for number in NUMBERS:
        pr = by_number[number]
        item = candidates.get(number, dict(head=pr['headRefOid'],
                                          old_head=pr['headRefOid']))
        sha(item['head'])
        require(item['old_head'] == pr['headRefOid'], 'Candidate old head differs')
        if number not in UNCHANGED:
            require(item['branch'] == pr['headRefName']
                    and item['base'] == pr['baseRefName'],
                    'Candidate branch/base differs from snapshot')
            sha(item['tree'])
            sha(item['parent'])
        mapping[number] = dict(item, url=pr['url'])
    expected_parents = {15: PARENT_15, 19: mapping[15]['head'],
                        16: mapping[19]['head'], 17: mapping[15]['head']}
    require(all(mapping[n]['parent'] == p for n, p in expected_parents.items()),
            'Candidate stack linkage differs')
    evidence = verify_context(context, args.context.resolve(), inputs,
                              args.snapshot, args.stack)
    bodies = {n: render(by_number[n], mapping, context, evidence) for n in NUMBERS}
    temporary = Path(tempfile.mkdtemp(prefix='.bodies-', dir=args.out.parent))
    try:
        metadata = []
        for number, body in bodies.items():
            pr = by_number[number]
            name = f'pr-{number}.md'
            (temporary / name).write_text(body)
            diff = difflib.unified_diff(
                pr['body'].splitlines(True), body.splitlines(True),
                fromfile=f'PR-{number}-snapshot', tofile=name)
            (temporary / f'pr-{number}.diff').write_text(''.join(diff))
            metadata.append(dict(number=number, url=pr['url'], title=pr['title'],
                                 base=pr['baseRefName'], branch=pr['headRefName'],
                                 isDraft=pr['isDraft'], head=mapping[number]['head'],
                                 old_head=pr['headRefOid'], body_file=name,
                                 body_sha256=digest(body.encode()),
                                 required_labels=['agent-assisted']))
        manifest = dict(schema='torch-performance-fix-bodies-v1',
                        created_utc=datetime.now(timezone.utc).isoformat(),
                        inputs_sha256=inputs, prs=metadata,
                        notice='Local review artifacts only. '
                        'No PR or label was modified.')
        (temporary / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        require(all(digest(Path(p).read_bytes()) == h for p, h in inputs.items()),
                'An input changed during rendering')
        require(not args.out.exists(), 'Output appeared during rendering')
        temporary.rename(args.out)
        print(json.dumps(dict(output=str(args.out.resolve()), prs=list(NUMBERS))))
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


if __name__ == '__main__':
    main()
