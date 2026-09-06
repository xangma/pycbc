"""Build the eleven Torch pages and verify the final comparison figure."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, default=Path('/private/tmp/pycbc-torch-inspiral-hotpaths-20260906'))
p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'docs-validation')
a = p.parse_args()
repo, out = a.root.resolve(), a.out.resolve()
out.mkdir(parents=True, exist_ok=False)
source = out / 'source'
source.mkdir()
pages = (
    'torch', 'torch_benchmark_details', 'torch_filtering',
    'torch_inspiral_reference',
    'torch_optimizations', 'torch_parity', 'torch_performance',
    'torch_runtime', 'torch_search', 'torch_testing', 'torch_workflows',
)
observed_pages = tuple(sorted(path.stem for path in (repo / 'docs').glob('torch*.rst')))
assert observed_pages == pages, (observed_pages, pages)
labels = {}
for path in (repo / 'docs').rglob('*.rst'):
    for label in re.findall(r'^\.\. _([^:\n]+):\s*$', path.read_text(errors='replace'), re.M):
        labels[label] = str(path.relative_to(repo))
refs, docs = set(), set()

def target(text):
    return text.rsplit('<', 1)[-1].rstrip('>') if '<' in text else text

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for name in pages:
    path = repo / 'docs' / (name + '.rst')
    shutil.copyfile(path, source / path.name)
    content = path.read_text()
    refs.update(target(value) for value in re.findall(r':ref:`([^`]+)`', content))
    docs.update(target(value) for value in re.findall(r':doc:`([^`]+)`', content))
assert refs <= labels.keys(), refs - labels.keys()
external_refs = sorted(r for r in refs if Path(labels[r]).stem not in pages)
external_docs = {}
for doc in sorted(docs - set(pages)):
    path = repo / 'docs' / (doc + '.rst')
    assert path.is_file(), f'Unresolved external doc {doc}'
    external_docs[doc] = dict(path=str(path), sha256=digest(path))
    stub = source / (doc + '.rst')
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text(':orphan:\n\n' + doc + '\n' + '=' * len(doc) + '\n\nExternal document exists in repository source; omitted from this focused build.\n')
collections = {'torch': ('original-vs-torch.png',)}
manifest_names = {'torch': 'manifest.json'}
folder = Path('images/torch')
shutil.copytree(repo / 'docs' / folder, source / folder)
expected_names = {name for names in collections.values() for name in names}
assert {path.name for path in (source / folder).glob('*.png')} == expected_names
assert {path.name for path in (source / folder).glob('*.json')} == set(manifest_names.values())
manifests = {}
for name in collections:
    manifest = json.loads((source / folder / manifest_names[name]).read_text())
    names = [figure['file'] for figure in manifest['figures']]
    assert len(names) == len(set(names)), f'Duplicate figures in {name}'
    assert set(names) == set(collections[name]), (name, names, collections[name])
    for figure in manifest['figures']:
        assert digest(source / folder / figure['file']) == figure['sha256'], figure['file']
    manifests[name] = manifest
(source / 'conf.py').write_text("project = 'PyCBC Torch documentation'\nextensions = []\nmaster_doc = 'index'\nhtml_theme = 'alabaster'\nexclude_patterns = []\nnitpick_ignore = " + repr([('std:ref', r) for r in external_refs]) + '\n')
(source / 'index.rst').write_text('PyCBC Torch documentation\n=========================\n\n.. toctree::\n   :maxdepth: 1\n\n' + ''.join('   ' + page + '\n' for page in pages if True))
command = [sys.executable, '-B', '-m', 'sphinx', '-E', '-a', '-n', '-W', '--keep-going', '-b', 'html', str(source), str(out / 'html')]
result = subprocess.run(command, text=True, capture_output=True)
(out / 'sphinx.log').write_text(result.stdout + result.stderr)
print(result.stdout + result.stderr)
result.check_returncode()
image_count = 0
for name, manifest in manifests.items():
    for figure in manifest['figures']:
        assert digest(out / 'html/_images' / figure['file']) == figure['sha256'], figure['file']
        image_count += 1
assert image_count == 1
expected_images = {figure for names in collections.values() for figure in names}
assert len(expected_images) == 1
assert {path.name for path in (out / 'html/_images').glob('*.png')} == expected_images
downloads = list((out / 'html/_downloads').glob('*/manifest.json'))
expected = {digest(source / folder / manifest_names[name]) for name in collections}
assert len(downloads) == len(expected) == 1
assert {digest(path) for path in downloads} == expected
receipt = dict(finished_utc=datetime.now(timezone.utc).isoformat(), sphinx_command=command,
               returncode=result.returncode, scope='Eleven Torch pages; external document targets verified and stubbed; full site not built',
               pages=pages, page_sha256={name: digest(repo / 'docs' / (name + '.rst')) for name in pages},
               manifest_sha256={name: digest(source / folder / manifest_names[name]) for name in collections},
               log_sha256=digest(out / 'sphinx.log'), external_labels={r: labels[r] for r in external_refs},
               external_documents=external_docs, figure_collections=collections,
               images=image_count, image_hashes_verified=True,
               downloadable_manifests_verified=True)
(out / 'docs-validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('Verified 11 Torch pages, one built image and one downloadable manifest.')
