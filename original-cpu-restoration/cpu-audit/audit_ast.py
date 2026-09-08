"""Read immutable Git blobs; emit normalized review evidence only beside this script."""
import ast
import difflib
import json
from pathlib import Path
import subprocess

REPO = Path('/Users/xangma/repos/pycbc')
OUT = Path(__file__).resolve().parent
BASE = '40e94792b3edf59f39b18b65102b28a4f74433a7'
CORRECTED = '66789ac4a7468094b0cc3ca1498a1de67e0311f6'
MAIN = '1d22031fd31c6e5bcb48a68fe11772e960bd406b'


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True)


class Normalize(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


def source(ref, path):
    return git('show', f'{ref}:{path}')


def tree(ref, path):
    return Normalize().visit(ast.parse(source(ref, path)))


def symbols(root):
    result = {}
    def walk(body, prefix=''):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result[prefix + node.name] = node
            elif isinstance(node, ast.ClassDef):
                walk(node.body, prefix + node.name + '.')
    walk(root.body)
    return result


def main():
    paths = git('diff', '--name-only', '--diff-filter=M', BASE, MAIN, '--', 'pycbc', 'bin').splitlines()
    inventory = {}
    for path in paths:
        if not path.endswith('.py') and not path.startswith('bin/'):
            continue
        old, corrected, new = (tree(ref, path) for ref in (BASE, CORRECTED, MAIN))
        a, b, c = map(symbols, (old, corrected, new))
        changes = []
        for name in sorted(a.keys() | c.keys()):
            if name not in a:
                changes.append({'symbol': name, 'status': 'added', 'main_line': c[name].lineno})
            elif name not in c:
                changes.append({'symbol': name, 'status': 'removed', 'base_line': a[name].lineno})
            elif ast.dump(a[name]) != ast.dump(c[name]):
                changes.append({'symbol': name, 'status': 'changed', 'base_line': a[name].lineno,
                                'main_line': c[name].lineno,
                                'known_correction_changes_symbol': ast.dump(a[name]) != ast.dump(b[name]),
                                'later_stack_changes_symbol': ast.dump(b[name]) != ast.dump(c[name])})
        inventory[path] = changes
        for label, left in [('baseline', old), ('post-corrections', corrected)]:
            diff = ''.join(difflib.unified_diff(
                ast.unparse(left).splitlines(True), ast.unparse(new).splitlines(True),
                fromfile=f'{label}:{path}', tofile=f'main:{path}'))
            (OUT / (path.replace('/', '__') + '.' + label + '.diff')).write_text(diff)
    (OUT / 'ast-inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps({p: [x['symbol'] for x in v if x['status'] != 'added']
                      for p, v in inventory.items()}, indent=2))


if __name__ == '__main__':
    main()
