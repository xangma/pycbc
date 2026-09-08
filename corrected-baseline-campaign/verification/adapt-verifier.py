"""Derive the corrected-source verifier from the previously reviewed verifier."""
import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / 'torch-baseline-final-20260908' / 'verify-results.py'
text = OLD.read_text()
old_sha = hashlib.sha256(OLD.read_bytes()).hexdigest()

def replace_node(name, replacement):
    global text
    node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    lines = text.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [replacement.rstrip() + '\n']
    text = ''.join(lines)

text = text.replace('original', 'corrected')
text = text.replace('corrected_manifest_unchanged', 'original_manifest_unchanged')
text = text.replace('40e94792b3edf59f39b18b65102b28a4f74433a7', '66789ac4a7468094b0cc3ca1498a1de67e0311f6')
text = text.replace('123e1fb3ef1b338cada636e71c3e9c7987002402', 'f582b6fd250d0b82612492979e01e645d5c07afc')
for old,new in {
'f9917b9ba994900e3e5707e882d57b03076bc01f89c4f37120c186a7d02c19bd':'4c10272822310a6f4e07b9068f014a164d64a420ee4a1778e2316b9993316a2f',
'08738a96432f127d3ea799a9714266de194318b0404692207c286b5c74621405':'78d4c5bb2bf45278a98a15689f25f1f1a10efac33f931a22cfe36e0247f4d944',
'ae3728c3c2e4365748c0cc3ab893b500526e0d7ee19b5c79afe565fd3d0dfb24':'c4f30a5edd4e1131d601dd266ee2f0bbfe48c538a47f015e286d58e99ceb2f07',
'74ba91a084b1a961d03d1f9d7ed78653858d9aa6784019b479ea88ae73271501':'f26b846eee2c87836335b3998c4a018681390f515340631249b09b15f7dadbbb',
}.items():
    assert old in text
    text = text.replace(old,new)
text = text.replace("def __init__(self, root):", "def __init__(self, root, qualification_only=False):")
text = text.replace('        self.inventory = {}', '        self.qualification_only = qualification_only\n        self.inventory = {}')
text = text.replace("        self.source_review = None", "        self.source_review = None\n        self.dependency_versions = {}")
text = text.replace("        self.pins = self.read('source-pins.json')", """        self.pins = self.read('source-pins.json')
        require(self.inventory['source-pins.json'] == '3745c9d356dbe2c162f7f57c681179e4f9119d001a269cb57b2089086ccaa219',
                'Source pins differ from reviewed qualification snapshot')""")
text = text.replace("build['commit'] == commit and build['status'] == '' and", "build['commit'] == commit and")
text = text.replace("""            require(build['command'] == [self.deps['executable'], 'setup.py', 'build_ext', '--inplace'],
                    'Build interpreter/command mismatch')""", """            references = {
                'corrected': ('original', '40e94792b3edf59f39b18b65102b28a4f74433a7',
                    '6c319b700a7673184eee2a91940054bec55945facc5e4458d39336ae409da451'),
                'proposed': ('proposed', '123e1fb3ef1b338cada636e71c3e9c7987002402',
                    '3d6592ea1bfc023cdf993f1f13121b1e505ee317ef3698b4b0bef698f2ae2bd6')}
            reference_name, reference_commit, reference_hash = references[name]
            require(build['method'] == 'Reused frozen binaries after exact native-source and binary-hash verification'
                    and build['reference_commit'] == reference_commit
                    and build['reference_build_sha256'] == reference_hash
                    and build['reference_source'] == '/home/xangma/pycbc-torch-baseline-final-20260908/' + reference_name,
                    'Reused native build provenance mismatch')""")
# The old post-acquisition diagnostic is tied to a different run. Do not transplant it.
start = text.index('        if any((self.root / name).exists() for name in DEPENDENCY_DIAGNOSTIC):')
end = text.index("        self.read('machine.json')",start)
text = text[:start] + """        review_path = Path(__file__).resolve().parent / 'source-review.json'
        review = strict_json(review_path.read_text())
        require(digest(review_path) == SOURCE_REVIEW_SHA256 and review['status'] == 'PASS'
                and review['source_pins_sha256'] == self.inventory['source-pins.json'],
                'Independent source review differs from frozen receipt')
        self.source_review = dict(sha256=SOURCE_REVIEW_SHA256, receipt=review)
        diagnostic_names = ('torch-environment-diagnostic.json', 'inspect-torch-environment.py')
        if not self.qualification_only and any((self.root / name).exists() for name in diagnostic_names):
            require(digest(self.file(diagnostic_names[1])) == DIAGNOSTIC_SCRIPT_SHA256,
                    'Unreviewed post-run dependency diagnostic script')
            require(self.continuation is not None, 'Diagnostic requires the completed timing continuation')
            self.dependency_reconciliation = dependency_diagnostic(
                self.read(diagnostic_names[0]), self.deps, self.timing_status,
                self.inventory['dependencies.json'], self.inventory['timing-status.json'],
                self.inventory[diagnostic_names[0]], DIAGNOSTIC_SCRIPT_SHA256)
""" + text[end:]
# Keep a mismatch explicit while still checking every worker against frozen runtime versions.
start = text.index('            if self.dependency_reconciliation is None:')
end = text.index('        else:\n            require(runtime',start)
text = text[:start] + """            require(runtime['torch_version'] == '2.13.0+cu130' and runtime['torch_cuda_version'] == '13.0',
                    'Worker Torch version differs from the qualification runtime contract')
            if self.dependency_reconciliation is not None:
                require(runtime['hostname'] == self.dependency_reconciliation['hostname'],
                        'Worker host differs from post-run diagnostic host')
            self.dependency_versions[case] = dict(metadata_torch=packages['torch'],
                worker_torch=runtime['torch_version'], worker_cuda=runtime['torch_cuda_version'],
                metadata_matches_runtime=packages['torch'] == runtime['torch_version'])
""" + text[end:]
text = text.replace("                'proposed trigger parity, exact conditioned strain/geometry, and exact used PSD bins pass. '",
                    "                'all corrected-baseline and proposed-backend trigger comparisons, exact conditioned strain/geometry, and exact used PSD bins pass. '")
text = text.replace("        self.timing_status = self.read('timing-status.json')\n", """        if self.qualification_only:
            self.timing_status = None
            self.continuation = dict(policy=policy, pins=pins, timing_status=None,
                                     initial_scientific_failure_preserved=True)
            return
        self.timing_status = self.read('timing-status.json')
""")
text = text.replace("    names = ['proposed-cpu-vs-' + arm for arm in ('torch-cpu', 'torch-cuda')]", "    names = [name for name, _, _ in PAIRS]")
# Preserve old functions for arithmetic checks but correct the first-stop order to this campaign.
text = text.replace("name for name, left, _ in PAIRS if left == 'proposed-cpu' and", "name for name, left, _ in PAIRS if")
text = text.replace("# The frozen initial controller stops at the first failing proposed pair.",
                    "# The frozen initial controller stops at the first failing pair in schedule order.")
# Verify four qualification cases independently while timings remain with the primary.
text = text.replace("        if self.continuation:\n            require(self.timing_status['completed']", "        if self.qualification_only:\n            expected_cases = expected_cases[:4]\n        elif self.continuation:\n            require(self.timing_status['completed']",1)
text = text.replace("        require(actual_cases == set(expected_cases), 'Expected exactly four qualifications and 16 timing cases')",
                    """        if self.qualification_only:
            require(set(expected_cases).issubset(actual_cases), 'Missing qualification case')
            require(all(name in set(expected_cases) or name.startswith('timing-') for name in actual_cases),
                    'Unexpected run directory')
        else:
            require(actual_cases == set(expected_cases), 'Expected exactly four qualifications and 16 timing cases')""")
text = text.replace("        timings = {arm: [] for arm in arms}", """        if self.qualification_only:
            self.check_case_order(expected_cases)
            for rel, value in self.inventory.items():
                require(digest(self.root / rel) == value, 'Evidence changed during verification: ' + rel)
            return dict(evidence_status='PASS_FOR_QUALIFICATION_WITH_LIMITATIONS',
                scientific_status='PASS' if all(v['conditioning']['pass'] and v['result']['status'] == 'pass'
                    for v in comparisons.values()) else 'FAIL',
                equal_output_speedup_eligible=False, timing_status='NOT_EVALUATED_PROVISIONAL_SNAPSHOT',
                comparisons=comparisons, continuation=self.continuation,
                trigger_counts={arm: self.loaded['qual-' + arm]['detectors']['H1']['count'] for arm in arms},
                timing_consistency={'allowances_seconds': TIMING_SLACK, 'cases': self.timing_checks},
                dependency_versions=self.dependency_versions,
                limitations=['Timing acquisition has not been evaluated; no performance verdict is available.',
                    'Setup distribution metadata reports Torch 2.1.1; worker runtime receipts report 2.13.0+cu130. '
                    'This acquisition has no current diagnostic binding that discrepancy to installation bytes.',
                    'Conditioned strain equality uses recorded digests and metadata. Strain and scientific input bytes '
                    'are not transferred; their identity is supported by acquisition receipts.',
                    'Native binary reuse is checked against prior receipts and unchanged source; native bytes are not transferred.'],
                source_content_verification=self.archived_files, independent_source_review_receipt=self.source_review,
                schedule=expected_cases, input_evidence_sha256=dict(sorted(self.inventory.items())))
        timings = {arm: [] for arm in arms}""")
text = text.replace("return dict(evidence_status='PASS', scientific_status=", "return dict(evidence_status='PASS_WITH_LIMITATIONS', scientific_status=")
text = text.replace("            dependency_reconciliation=self.dependency_reconciliation,", """            dependency_versions=self.dependency_versions,
            dependency_reconciliation=self.dependency_reconciliation,
            dependency_metadata_status=(self.dependency_reconciliation['status'] if self.dependency_reconciliation
                else 'MISMATCH_UNRECONCILED_IN_THIS_ACQUISITION'),""")
# Add shared ordering check for provisional qualifications; full-run check remains intact.
needle = '    def run(self):'
text = text.replace(needle, """    def check_case_order(self, cases):
        previous_end = None
        for case in cases:
            rec = self.receipts[case]
            start, end = [dt.datetime.fromisoformat(rec[k]) for k in ('started_utc', 'finished_utc')]
            require(start.tzinfo is not None and end.tzinfo is not None and end > start
                    and (previous_end is None or start >= previous_end), 'Case order/timestamps disagree')
            require(rec['parent_pid'] == self.status['pid'] and rec['hostname'] == 'len',
                    'Qualification host or parent controller mismatch')
            previous_end = end

""" + needle)
text = text.replace("            self.checker.check(runtime[key], env, bank=key == 'at_first_bank', scheme=scheme)", """            self.checker.check(runtime[key], env, bank=key == 'at_first_bank', scheme=scheme)
            require(runtime[key]['torch_imported'] is (source_name == 'proposed'),
                    'Torch import state disagrees with the selected scheme')""")
# Retain relationship checks; bind the new script and this acquisition's JSON bytes.
text = text.replace('def dependency_diagnostic(diag, dependencies, timing_status, dependency_hash, status_hash):',
                    'def dependency_diagnostic(diag, dependencies, timing_status, dependency_hash, status_hash, diagnostic_hash, script_hash):')
text = text.replace("    require(diag['schema_version'] == 1 and", "    require(valid_hash(diagnostic_hash) and script_hash == DIAGNOSTIC_SCRIPT_SHA256,\n            'Invalid post-run diagnostic artifact identity')\n    require(diag['schema_version'] == 1 and")
text = text.replace("diagnostic_sha256=DEPENDENCY_DIAGNOSTIC['torch-environment-diagnostic.json']", 'diagnostic_sha256=diagnostic_hash')
text = text.replace("diagnostic_script_sha256=DEPENDENCY_DIAGNOSTIC['inspect-torch-environment.py']", 'diagnostic_script_sha256=script_hash')
start = text.index('DEPENDENCY_DIAGNOSTIC = {')
end = text.index('TIMING_SLACK = {',start)
text = text[:start] + "DIAGNOSTIC_SCRIPT_SHA256 = 'b98136e61bbcca021250b2fdbde2f59385aaf2d2e279b17e2c806159d2e258fe'\n" + text[end:]
text = text.replace("dependency_diagnostic(value, deps, timing, 'a' * 64, 'b' * 64)",
                    "dependency_diagnostic(value, deps, timing, 'a' * 64, 'b' * 64, 'd' * 64, DIAGNOSTIC_SCRIPT_SHA256)")
text = text.replace("    proposed = {'proposed-cpu-vs-' + arm: dict(result=trigger, conditioning=excluded_condition)\n                for arm in ('torch-cpu', 'torch-cuda')}",
                    "    proposed = {name: dict(result=trigger, conditioning=excluded_condition) for name, _, _ in PAIRS}")
text = text.replace("with tempfile.TemporaryDirectory(prefix='pycbc-offline-verifier-selftest-') as tmp:",
                    "with tempfile.TemporaryDirectory(prefix='selftest-', dir=Path(__file__).resolve().parent) as tmp:")
text = text.replace("    parser.add_argument('--self-test', action='store_true')", "    parser.add_argument('--self-test', action='store_true')\n    parser.add_argument('--qualification-only', action='store_true', help='Evaluate the four qualifications; never certify timings')")
text = text.replace("report.update(Verifier(root).run())", "report.update(Verifier(root, qualification_only=args.qualification_only).run())")
text = text.replace("    require(not output.exists(), 'Refusing to overwrite output')", """    require(not output.exists(), 'Refusing to overwrite output')
    require(output.is_relative_to(Path(__file__).resolve().parent), 'Output must remain within the verifier folder')""")
text = text.replace("Exit 0 means complete, consistent evidence, even when scientific_status is FAIL.",
                    "Exit 0 means the requested evidence scope is consistent, even when scientific_status is FAIL.\nQualification-only mode is explicitly provisional. Dependency metadata mismatch remains disclosed.")
source_review_sha = hashlib.sha256((HERE/'source-review.json').read_bytes()).hexdigest()
text = text.replace('a5c0c8ec2a5e1e9e352294f2265c8ca0d8fca869f1601e0722e5f52d9b67693f',source_review_sha)
text = text.replace('import argparse\n', "# Derived from prior verifier SHA256: " + old_sha + "\nimport argparse\n")
ast.parse(text)
(HERE/'verify-results.py').write_text(text)
print('Created',HERE/'verify-results.py')
