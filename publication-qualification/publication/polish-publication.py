"""Freeze narrowly reviewed lint cleanup on the four qualified candidates."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

PUB = Path(__file__).resolve().parent
ROOT = Path('/private/tmp/pycbc-torch-performance-fix-20260906-assembly')
COMMON = ['pycbc/fft/torchfft.py', 'pycbc/strain/strain.py',
          'pycbc/waveform/decompress_torch.py', 'pycbc/waveform/taylorf2_torch.py']
EXTRA = ['pycbc/fft/__init__.py', 'pycbc/fft/cufft.py']


def git(root, *args, data=None):
    return subprocess.run(['git', '-C', str(root), *args], input=data,
                          stdout=subprocess.PIPE, check=True).stdout


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalized(data, path):
    tree = ast.parse(data)
    if path == 'pycbc/strain/strain.py':
        moves = [n for n in tree.body if isinstance(n, ast.ImportFrom)
                 and n.module == 'pycbc' and len(n.names) == 1
                 and n.names[0].name == 'scheme' and n.names[0].asname == '_scheme']
        assert len(moves) == 1
        tree.body.remove(moves[0])
        tree.body.append(moves[0])
    if path == 'pycbc/fft/__init__.py':
        expanded = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                for name in node.names:
                    assert name.asname in (None, name.name)
                    expanded.append(ast.ImportFrom(module=node.module,
                        names=[ast.alias(name=name.name, asname=None)], level=node.level))
            else:
                expanded.append(node)
        tree.body = expanded
    if path == 'pycbc/fft/cufft.py':
        raises = [n for n in ast.walk(tree) if isinstance(n, ast.Raise)
                  and isinstance(n.exc, ast.Call) and isinstance(n.exc.func, ast.Name)
                  and n.exc.func.id == 'ImportError' and len(n.exc.args) == 1
                  and isinstance(n.exc.args[0], ast.Constant)
                  and n.exc.args[0].value == 'Unable to import skcuda.fft; try direct import to get full traceback']
        assert len(raises) == 1
        cause = raises[0].cause
        assert cause is None or (isinstance(cause, ast.Constant) and cause.value is None)
        raises[0].cause = None
    return ast.dump(tree, include_attributes=False)


def validate_delta(root, before, after, number):
    expected = sorted(COMMON + (EXTRA if number == 19 else []))
    actual = git(root, 'diff', '--name-only', before, after).decode().splitlines()
    assert actual == expected, (number, actual)
    result = {}
    for path in expected:
        old = git(root, 'show', before + ':' + path)
        new = git(root, 'show', after + ':' + path)
        assert normalized(old, path) == normalized(new, path), (number, path)
        modes = [git(root, 'ls-tree', ref, '--', path).split()[0] for ref in (before, after)]
        assert modes == [b'100644', b'100644']
        result[path] = dict(before_sha256=sha(old), after_sha256=sha(new),
                           normalized_ast_sha256=sha(normalized(new, path).encode()))
    return result


def write(name, value):
    with (PUB / name).open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def main():
    spec = importlib.util.spec_from_file_location('publisher', PUB / 'publish.py')
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    files = dict(restack_plan=PUB / 'frozen-plan.json', candidate_stack=PUB / 'candidate-stack.json',
                 restack_execution=PUB / 'restack-execution.json', restack_script=PUB / 'restack.py',
                 snapshot=PUB / 'prs-current-v6.json')
    publisher.candidate_checks(ROOT, files)
    stack = json.loads(files['candidate_stack'].read_bytes())
    assert not (PUB / 'candidate-stack-final.json').exists()
    rows, proof = [], []
    for old in stack['candidates']:
        n = old['number']
        work = Path('/private/tmp/pycbc-torch-publication-polish-20260906' + ('' if n == 15 else '-' + str(n)))
        assert git(work, 'rev-parse', 'HEAD').decode().strip() == old['head']
        paths = sorted(COMMON + (EXTRA if n == 19 else []))
        assert git(work, 'diff', '--name-only').decode().splitlines() == paths
        assert not git(work, 'diff', '--cached')
        assert not git(work, 'ls-files', '--others', '--exclude-standard')
        git(work, 'add', '--', *paths)
        tree = git(work, 'write-tree').decode().strip()
        details = validate_delta(ROOT, old['head'], tree, n)
        parent = old['parent'] if n == 15 else next(r['head'] for r in rows if r['branch'] == old['base'])
        subject = git(ROOT, 'log', '-1', '--format=%s', old['head'])
        head = git(ROOT, 'commit-tree', tree, '-p', parent, data=subject).decode().strip()
        branch = old['candidate_branch'] + '-final'
        git(ROOT, 'update-ref', 'refs/heads/' + branch, head, '0' * 40)
        row = dict(old, head=head, parent=parent, tree=tree, candidate_branch=branch)
        rows.append(row)
        proof.append(dict(number=n, before_head=old['head'], after_head=head,
                          before_tree=old['tree'], after_tree=tree, files=details))
    review = dict(schema='torch-publication-polish-v1', reviewed=True, rows=proof,
                  script_sha256=sha(Path(__file__).read_bytes()),
                  prior_inputs={k: dict(path=str(p), sha256=sha(p.read_bytes())) for k, p in files.items()})
    write('polish-proof-v6.json', review)
    info = dict(path=str(PUB / 'polish-proof-v6.json'), sha256=sha((PUB / 'polish-proof-v6.json').read_bytes()),
                script=str(Path(__file__)), script_sha256=review['script_sha256'])
    final = dict(stack, candidates=rows, polish=info)
    write('candidate-stack-final.json', final)
    receipt = json.loads(files['restack_execution'].read_bytes())
    write('restack-execution-final.json', dict(receipt, candidates=rows, polish=info))
    git(ROOT, 'bundle', 'create', str(PUB / 'publication-final.bundle'),
        *['refs/heads/' + r['candidate_branch'] for r in rows], '^dfd42bf76766cadca0eecf609a1eaeac73534676')
    git(ROOT, 'bundle', 'verify', str(PUB / 'publication-final.bundle'))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
