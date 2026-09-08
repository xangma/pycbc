"""Generate checkout-specific metadata using its unchanged setup.py function."""
import ast
import json
from pathlib import Path
import subprocess
import sys

source = Path.cwd()
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert head == sys.argv[1]
sys.path.insert(0, str(source))
tree = ast.parse((source / 'setup.py').read_text())
function, = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == 'get_version_info']
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source / 'setup.py'), 'exec'), namespace)
namespace['get_version_info']()
generated = ast.parse((source / 'pycbc/version.py').read_text())
values = {node.targets[0].id: ast.literal_eval(node.value) for node in generated.body
          if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
assert values['git_hash'] == head, values
assert values['git_status'] == 'CLEAN: All modifications committed', values
print(json.dumps({'source': str(source), 'generated_git_hash': values['git_hash']}))
